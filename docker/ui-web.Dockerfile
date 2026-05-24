# Production build for the React audit console (rie-ui-web).
FROM node:22-alpine AS builder

WORKDIR /app
COPY packages/rie-ui-web/package.json packages/rie-ui-web/package-lock.json* ./
RUN npm install

COPY packages/rie-ui-web/ ./
ARG VITE_API_BASE=
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/ui-web.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
