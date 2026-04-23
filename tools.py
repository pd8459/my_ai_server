# tools.py
import math

def calculate_dog_calories(weight: float, activity_level: str):
    """
    강아지의 몸무게와 활동 수준을 바탕으로 하루 권장 칼로리를 계산합니다.
    - weight: 강아지 몸무게 (kg)
    - activity_level: 'low', 'normal', 'high' 중 하나
    
    """
    rer = 70 * math.pow(weight, 0.75)
    
    factors = {"low": 1.2, "normal": 1.6, "high": 2.0}
    factor = factors.get(activity_level, 1.6)
    daily_calories = rer * factor
    
    return {
        "daily_calories": round(daily_calories, 2),
        "message": f"권장 칼로리는 하루 약 {round(daily_calories, 2)}kcal입니다."
    }

