#Initial setup:
python3 -m venv .venv
Linux: source .venv/bin/activate /// Windows(Visual Studio): .venv\Scripts\activate

#Start server:
uvicorn main:app --reload

http://127.0.0.1:8000/docs

#Free host on internet (random domain each time):
ngrok http 8000