# build environment
FROM node:13.12.0-alpine as build
COPY package.json ./
COPY yarn.lock ./
RUN yarn install
RUN yarn global add react-scripts@3.4.1
COPY . .
RUN yarn run build

# production environment
FROM nginx:stable-alpine
COPY --from=build build /usr/share/nginx/html
CMD ["nginx", "-g", "daemon off;"]