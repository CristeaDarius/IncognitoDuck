Project made in 24h for the Hackathon 'KreativHack OUTech', the theme has 'AI for Cultural Heritage', and it was held on September 17–19, 2025. 

#Initial setup (Linux):
python3 -m venv .venv
source .venv/bin/activate

#Start server:
uvicorn main:app --reload

http://127.0.0.1:8000/docs

#Free host on internet (random domain each time):
ngrok http 8000
