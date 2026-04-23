import io
import os
import json
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from tools import calculate_dog_calories
from data.breeds import BREEDS_KR

load_dotenv()
app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

image_model = models.resnet50(weights=None)
num_breeds = 120
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, num_breeds)
image_model.load_state_dict(torch.load('models/dog_breed_stage1.pt', map_location=device))
image_model.to(device)
image_model.eval()

chatbot_model = SentenceTransformer('models/puppytalk-chatbot-model').to(device)

qna_data = {"강아지 처음 키우는데 뭐가 필요해?": "사료, 밥그릇, 물그릇, 배변패드, 안전한 장난감, 목줄과 같은 기본적인 용품들이 필요해요.", "강아지 사회화는 어떻게 시켜?": "생후 3주에서 12주 사이 '사회화 시기'에 다른 강아지나 새로운 사람들을 긍정적으로 만나게 해주는 것이 중요해요.", "예방접종은 언제부터 해야 돼?": "보통 생후 6~8주부터 시작하며, 동물병원 의사 선생님과 상담하여 정확한 접종 일정을 잡는 것이 가장 안전해요.", "배변 훈련은 어떻게 시작해?": "강아지가 자고 일어났을 때, 밥을 먹고 난 후에 지정된 배변 장소로 데려가서 성공하면 칭찬과 간식으로 보상해주세요.", "우리 강아지가 자꾸 짖는데 왜 그래?": "심심해서 관심을 끌고 싶거나, 무언가 경계하거나, 불안함을 느낄 때 짖을 수 있어요. 상황을 잘 살펴보는 것이 중요해요.", "강아지 혼자 둬도 괜찮아?": "강아지는 무리 동물이어서 혼자 있는 것을 힘들어할 수 있어요. 짧은 시간부터 시작해서 점차 혼자 있는 시간을 늘려가는 훈련이 필요해요.", "사료는 얼마나 줘야 해?": "사료 포장지에 적힌 몸무게별 권장 급여량을 따르는 것이 기본이지만, 강아지의 활동량이나 건강 상태에 따라 조절이 필요해요.", "강아지가 먹으면 안 되는 음식이 뭐야?": "초콜릿, 양파, 마늘, 포도, 마카다미아 등은 강아지에게 매우 위험하니 절대 주면 안 돼요.", "'앉아' 훈련은 어떻게 가르쳐?": "간식을 강아지 코앞에 댔다가 천천히 머리 위로 올리면 자연스럽게 앉게 돼요. 앉는 순간 '앉아!'라고 말하며 간식을 주세요.", "산책은 매일 해야 돼?": "네, 산책은 강아지의 스트레스 해소와 에너지 발산, 사회성 발달에 필수적이므로 하루 1~2회 정도 해주는 것이 좋아요."}

qna_embeddings = chatbot_model.encode(list(qna_data.keys()), convert_to_tensor=True).to(device)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ChatRequest(BaseModel):
    question: str

def predict_logic(contents: bytes):
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, 1).item()
        confidence = probs[0][pred_idx].item()
    name = BREEDS_KR[pred_idx] if pred_idx < len(BREEDS_KR) else "알 수 없음"
    return {"breed": {"id": pred_idx, "name": name, "confidence": round(confidence, 3)}}

@app.post("/predict/dog-breed")
async def predict_dog_breed(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return await run_in_threadpool(predict_logic, contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_emb = chatbot_model.encode(request.question, convert_to_tensor=True).to(device)
        cos_scores = util.cos_sim(user_emb, qna_embeddings)
        top_results = cos_scores.topk(k=2)
        contexts = [list(qna_data.values())[idx] for idx in top_results.indices[0]]
        retrieved_knowledge = "\n".join(contexts)

        prompt = f"""당신은 퍼피톡의 건강 매니저입니다.
context:
{retrieved_knowledge}

question:
{request.question}

필요하면 JSON:
{{"action":"calculate_calories","weight":10,"activity":"normal"}}"""

        completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        ai_response = completion.choices[0].message.content

        try:
            tool_data = json.loads(ai_response)
            if tool_data.get("action") == "calculate_calories":
                calc_result = calculate_dog_calories(tool_data["weight"], tool_data["activity"])
                final_prompt = f"""{request.question}
{calc_result['message']}
3줄 이내로 핵심만 답"""
                final_completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": final_prompt}], temperature=0.1)
                return {"answer": final_completion.choices[0].message.content}
        except:
            pass

        return {"answer": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
