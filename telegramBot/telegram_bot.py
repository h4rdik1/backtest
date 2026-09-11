# telegram_bot.py — Main entry point. Just wires everything together and starts the engine.

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from config import TOKEN
from handlers import start_command, get_id_command, scan_command, handle_text, handle_photo


def main():
    print("Bot is starting up...")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",  start_command))
    app.add_handler(CommandHandler("id",     get_id_command))
    app.add_handler(CommandHandler("scan",   scan_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is listening for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
