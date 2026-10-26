# Stage 1 builds the static site with Node. Stage 2 serves it with nginx, so Node is not shipped in
# the final image.
FROM node:22-alpine AS build
WORKDIR /app
# Lockfile first, so the npm install layer stays cached until the dependencies change.
COPY package.json package-lock.json ./
# npm ci installs exactly what package-lock.json pins, which makes builds reproducible.
RUN npm ci --no-audit --no-fund
COPY . .
# Leave empty to call /api on the same origin (proxied by nginx below).
# Vite inlines VITE_* values at build time, so changing this later means rebuilding the image.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Stage 2: runtime image.
FROM nginx:1.27-alpine
# BACKEND_URL is substituted into the nginx config at container start.
# The official nginx image runs envsubst over /etc/nginx/templates/*.template at start-up and writes
# the result to /etc/nginx/conf.d/, which is how ${PORT} and ${BACKEND_URL} reach the config.
ENV BACKEND_URL=http://backend:8000 PORT=80
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
