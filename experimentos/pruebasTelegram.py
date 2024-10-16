import telegram
import time
import asyncio

# Token de bot y chat ID
TOKEN = "7386977030:AAG15A9-JKyUzjrwVBFeAlXykC48QsQW_yk"
CHAT_ID = "7483893498"

async def send_telegram_message(message):
    bot = telegram.Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)

async def main():
    print("Ejecutando el código...")
    await asyncio.sleep(10)  # Simula la ejecución con asyncio

    # Enviar el mensaje
    await send_telegram_message("¡Tu código ha terminado de ejecutarse!")
    print("Mensaje enviado por Telegram.")

# Ejecutar la simulación y enviar el mensaje
if __name__ == "__main__":
    asyncio.run(main())





