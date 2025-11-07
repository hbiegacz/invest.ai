FROM node:20-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
COPY . .

ENV CHOKIDAR_USEPOLLING=true
ENV WATCHPACK_POLLING=true
ENV HOST=0.0.0.0
ENV NPM_SCRIPT=dev

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5173
ENTRYPOINT ["/entrypoint.sh"]
