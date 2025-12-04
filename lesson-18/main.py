# ================== TNG DEEPSEEK R1T CHIMERA CHATBOT (Terminal, Free) ==================
# Model: tngtech/deepseek-r1t-chimera:free (OpenRouter orqali)
# Token: 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
# Req: pip install openai

import os
from openai import OpenAI

# API sozlamalari
API_KEY = "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"
MODEL_NAME = "tngtech/deepseek-r1t-chimera:free"
BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Suhbat holati (kontekst saqlash uchun)
conversation_history = [
    {"role": "system", "content": "Sen o'zbek tilida aqlli, do'stona va foydali chatbot-san. Har doim o'zbekcha javob ber, qisqa va aniq bo'l. Hozirgi sana: 2025-yil 28-noyabr."}
]

def get_response(user_message: str) -> str:
    # Kontekstga qo'shish
    conversation_history.append({"role": "user", "content": user_message})
    
    # API chaqirish
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            temperature=0.7,  # Ijodkorlik darajasi
            max_tokens=800,  # Javob uzunligi
            stream=False  # Terminal uchun to'liq javob
        )
        answer = response.choices[0].message.content.strip()
        
        # Kontekstga javobni qo'shish
        conversation_history.append({"role": "assistant", "content": answer})
        
        # Kontekstni cheklash (oxirgi 10 ta xabar, token tejash uchun)
        if len(conversation_history) > 12:  # System + 5 pair
            conversation_history[:] = [conversation_history[0]] + conversation_history[-10:]
        
        return answer
    except Exception as e:
        return f"Xato: {str(e)}. Token yoki internetni tekshiring. (Status: {getattr(e, 'response', 'N/A')}')"

# Terminal loop
def main():
    print("🚀 TNG: DeepSeek R1T Chimera Chatbot (Free Tier)")
    print("Model: 671B MoE | Bepul limit: Kunlik ~1M token")
    print("Yozing (chiqish: 'quit' yoki 'exit'):\n")
    
    while True:
        try:
            user_input = input("Siz: ").strip()
            if user_input.lower() in ['quit', 'exit', 'chiqish']:
                print("Bot: Xayr! Yana suhbatlashamiz. 👋")
                break
            if not user_input:
                continue
            
            print("Bot: ", end="", flush=True)
            reply = get_response(user_input)
            print(reply + "\n")
            
        except KeyboardInterrupt:
            print("\nBot: Xayr! Suhbat tugadi.")
            break
        except Exception as e:
            print(f"\nXato: {e}")

if __name__ == "__main__":
    main()