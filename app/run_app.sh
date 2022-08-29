$HOME/.poetry/bin/poetry shell
/fs/clip-quiz/amao/caddy run &
uvicorn backend.server:app &

cd app
yarn start &

