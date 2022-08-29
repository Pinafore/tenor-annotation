# pull official base image
FROM node:13.12.0-alpine

# set working directory
# WORKDIR /app
WORKDIR /frontend

# add `/app/node_modules/.bin` to $PATH
# ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
# COPY package-lock.json ./
COPY yarn.lock ./

# RUN npm install --silent
# RUN npm install react-scripts@3.4.1 -g --silent
# RUN npm install
# RUN npm install react-scripts@3.4.1 -g
RUN yarn install
# RUN yarn install react-scripts@3.4.1 -g
RUN yarn global add react-scripts@3.4.1

# add app
# COPY . ./
COPY . ./
# WORKDIR "./frontend"

# start app
# CMD ["npm", "start"]
CMD ["yarn", "start"]