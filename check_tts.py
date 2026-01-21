import asyncio
import edge_tts

TEXT = "ውል ማለት በሁለት ወይም ከሁለት በላይ በሆኑ ሰዎች መካከል የሚደረግ ስምምነት ነው።"
VOICE = "am-ET-MekdesNeural" # Or "am-ET-MekdesNeural"
OUTPUT_FILE = "legal_sample.mp3"

async def generate():
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(generate())