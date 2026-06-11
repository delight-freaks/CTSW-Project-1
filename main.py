import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# 1. 클라이언트 설정 (API는 제 Claude 임의 API를 사용하게끔 하였습니다.)
client = anthropic.Anthropic(
    api_key="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"
)

@app.get("/")
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/analyze")
async def analyze_keystrokes(request: Request):
    data = await request.json()
    text = data.get("text", "")
    speed = data.get("speed", 0)

    # 2. 감정 분석 로직
    emotion = "불안함" if speed > 5 else ("무기력함" if speed < 1 else "안정적")

    # 3. Claude 호출
    response = client.messages.create(
        model="claude-sonnet-4-6", 
        max_tokens=1024,
        system="당신은 심리 상담사입니다. 사용자의 감정 상태를 분석하고 공감하며 대화하세요.",
        messages=[{"role": "user", "content": f"사용자 상태: {emotion}. 내용: {text}"}]
    )
    
    return {"reply": response.content[0].text, "emotion": emotion}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)