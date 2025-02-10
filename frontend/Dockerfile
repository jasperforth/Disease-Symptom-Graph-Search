# frontend/Dockerfile
# Stage1: Build the React app
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./
RUN npm install

# Copy the app
COPY . .

# Build the production version of the app
RUN npm run build

# Stage2: Serve the React app with Nginx
FROM nginx:alpine

# Remove the default Nginx configuration file
RUN rm /usr/share/nginx/html/*

# Copy the build output from the first stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]