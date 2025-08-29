import io
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer, util


image_model = models.resnet50()
num_breeds = 120 
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, num_breeds)
image_model_path = 'models/dog_breed_stage1.pt'  
image_model.load_state_dict(torch.load(image_model_path, map_location=torch.device('cpu')))
image_model.eval()

idx_to_breed_kr = {
    0: '아펜핀셔', 1: '아프간 하운드', 2: '아프리카 헌팅 도그', 3: '아이리든', 
    4: '아메리칸 스태퍼드셔 테리어', 5: '아펜첼러', 6: '오스트레일리안 테리어',
    7: '바센지', 8: '배셋', 9: '비글', 10: '베들링턴 테리어', 
    11: '버니즈 마운틴 도그', 12: '블랙 앤 탄 쿤하운드', 13: '블렌하임 스패니얼', 
    14: '블러드하운드', 15: '블루틱', 16: '보더 콜리', 17: '보더 테리어', 
    18: '보르조이', 19: '보스턴 불', 20: '부비에 데 플랑드르', 21: '복서',
    22: '브라방송 그리펀', 23: '브리아드', 24: '브리타니 스패니얼', 25: '불 마스티프',
    26: '케언', 27: '카디건', 28: '체서피크 베이 리트리버', 29: '치와와', 
    30: '차우', 31: '클럼버', 32: '코커 스패니얼', 33: '콜리', 34: '컬리 코티드 리트리버',
    35: '댄디 딘몬트', 36: '다올', 37: '딩고', 38: '도베르만', 39: '잉글리시 폭스하운드',
    40: '잉글리시 세터', 41: '잉글리시 스프링거', 42: '엔텔부허', 43: '에스키모 도그',
    44: '플랫 코티드 리트리버', 45: '프렌치 불독', 46: '저먼 셰퍼드', 
    47: '저먼 숏헤어드 포인터', 48: '자이언트 슈나우저', 49: '골든 리트리버',
    50: '고든 세터', 51: '그레이트 데인', 52: '그레이트 피레니즈', 
    53: '그레이터 스위스 마운틴 도그', 54: '그로넨달', 55: '이비잔 하운드',
    56: '아이리시 세터', 57: '아이리시 테리어', 58: '아이리시 워터 스패니얼',
    59: '아이리시 울프하운드', 60: '이탈리안 그레이하운드', 61: '재패니즈 스패니얼',
    62: '키숑', 63: '켈피', 64: '케리 블루 테리어', 65: '코몬도르', 
    66: '쿠바즈', 67: '래브라도 리트리버', 68: '레이클랜드 테리어', 69: '레온버그', 
    70: '라사', 71: '말라뮤트', 72: '말리노이즈', 73: '말티즈', 74: '멕시칸 헤어리스', 
    75: '미니어처 핀셔', 76: '미니어처 푸들', 77: '미니어처 슈나우저', 78: '뉴펀들랜드',
    79: '노퍽 테리어', 80: '노르웨이 엘크하운드', 81: '노리치 테리어', 
    82: '올드 잉글리시 쉽도그', 83: '오터하운드', 84: '파필롱', 85: '페키니즈',
    86: '펨브로크', 87: '포메라니안', 88: '퍼그', 89: '레드본', 90: '로드지시안 리지백',
    91: '로트와일러', 92: '세인트 버나드', 93: '살루키', 94: '사모예드', 95: '쉽퍼케',
    96: '스코치 테리어', 97: '스코티시 디어하운드', 98: '실리햄 테리어', 
    99: '셰틀랜드 쉽도그', 100: '시즈', 101: '시베리안 허스키', 102: '실키 테리어',
    103: '소프트 코티드 휘튼 테리어', 104: '스태퍼드셔 불테리어', 105: '스탠다드 푸들',
    106: '스탠다드 슈나우저', 107: '서섹스 스패니얼', 108: '티베탄 마스티프', 
    109: '티베탄 테리어', 110: '토이 푸들', 111: '토이 테리어', 112: '비즐라', 
    113: '워커 하운드', 114: '와이마라너', 115: '웰시 스프링거 스패니얼', 
    116: '웨스트 하이랜드 화이트 테리어', 117: '휘핏', 118: '와이어 헤어드 폭스 테리어',
    119: '요크셔 테리어'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

chatbot_model_path = 'models/puppytalk-chatbot-model'
chatbot_model = SentenceTransformer(chatbot_model_path)

qna_data = {
    "강아지 처음 키우는데 뭐가 필요해?": "사료, 밥그릇, 물그릇, 배변패드, 안전한 장난감, 목줄과 같은 기본적인 용품들이 필요해요.",
    "강아지 사회화는 어떻게 시켜?": "생후 3주에서 12주 사이 '사회화 시기'에 다른 강아지나 새로운 사람들을 긍정적으로 만나게 해주는 것이 중요해요.",
    "예방접종은 언제부터 해야 돼?": "보통 생후 6~8주부터 시작하며, 동물병원 의사 선생님과 상담하여 정확한 접종 일정을 잡는 것이 가장 안전해요.",
    "배변 훈련은 어떻게 시작해?": "강아지가 자고 일어났을 때, 밥을 먹고 난 후에 지정된 배변 장소로 데려가서 성공하면 칭찬과 간식으로 보상해주세요.",
    "우리 강아지가 자꾸 짖는데 왜 그래?": "심심해서 관심을 끌고 싶거나, 무언가 경계하거나, 불안함을 느낄 때 짖을 수 있어요. 상황을 잘 살펴보는 것이 중요해요.",
    "강아지 혼자 둬도 괜찮아?": "강아지는 무리 동물이어서 혼자 있는 것을 힘들어할 수 있어요. 짧은 시간부터 시작해서 점차 혼자 있는 시간을 늘려가는 훈련이 필요해요.",
    "사료는 얼마나 줘야 해?": "사료 포장지에 적힌 몸무게별 권장 급여량을 따르는 것이 기본이지만, 강아지의 활동량이나 건강 상태에 따라 조절이 필요해요.",
    "강아지가 먹으면 안 되는 음식이 뭐야?": "초콜릿, 양파, 마늘, 포도, 마카다미아 등은 강아지에게 매우 위험하니 절대 주면 안 돼요.",
    "'앉아' 훈련은 어떻게 가르쳐?": "간식을 강아지 코앞에 댔다가 천천히 머리 위로 올리면 자연스럽게 앉게 돼요. 앉는 순간 '앉아!'라고 말하며 간식을 주세요.",
    "산책은 매일 해야 돼?": "네, 산책은 강아지의 스트레스 해소와 에너지 발산, 사회성 발달에 필수적이므로 하루 1~2회 정도 해주는 것이 좋아요."
}

qna_embeddings = chatbot_model.encode(list(qna_data.keys()), convert_to_tensor=True)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/predict/dog-breed")
async def predict_dog_breed(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        pred_idx = torch.argmax(outputs, 1).item()
    return {"breed": idx_to_breed_kr[pred_idx]}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_emb = chatbot_model.encode(request.question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_emb, qna_embeddings)
    best_idx = cos_scores.argmax().item()
    return {"answer": qna_data[list(qna_data.keys())[best_idx]]}
