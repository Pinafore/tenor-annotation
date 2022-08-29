# backend: poetry, create tables, fastapi

# FROM python:3.9 as requirements-stage
FROM python:3.7 as requirements-stage

# 
WORKDIR /tmp

# 
RUN pip install poetry

# 
COPY ./pyproject.toml ./poetry.lock* /tmp/

# 
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# 
FROM python:3.7

# 
WORKDIR /code

# 
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

# install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# temp fix. TODO: add to requirements
RUN pip install aiofiles pytest

RUN python -m spacy download en_core_web_sm

# copy files to container
# COPY ./backend /code/backend
COPY ./ /code
# WORKDIR /code/backend/data
# WORKDIR /code

# create tables in database
# RUN python -m backend.helper_scripts.create_tables
# RUN python -m backend.create_tables

# 
# CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "81", "--reload"]
