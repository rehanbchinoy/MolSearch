# MolSearch Deployment Guide

This guide covers how to deploy MolSearch to various platforms.

## Local Development

### Prerequisites
- Python 3.8+
- RDKit
- Streamlit

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`

## Streamlit Cloud Deployment

### Steps
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

### Configuration
- **Python version**: 3.8+
- **Main file**: `app.py`
- **Requirements**: `requirements.txt`

## Heroku Deployment

### Steps
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t molsearch .
docker run -p 8501:8501 molsearch
```

## Environment Variables

Set these environment variables for production:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true)

## Performance Optimization

### For Production
1. **Enable caching**: Use `@st.cache_data` and `@st.cache_resource`
2. **Optimize database**: Use connection pooling
3. **Load balancing**: Use multiple instances behind a load balancer
4. **CDN**: Serve static assets through a CDN

### Monitoring
- Use Streamlit's built-in monitoring
- Set up logging to track usage
- Monitor database performance

## Security Considerations

1. **Input validation**: All SMILES strings are validated
2. **Rate limiting**: Implement rate limiting for API endpoints
3. **HTTPS**: Always use HTTPS in production
4. **Environment variables**: Store sensitive data in environment variables

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change the port in the deployment command
2. **Memory issues**: Increase memory allocation for large datasets
3. **Database errors**: Check database connection and permissions

### Logs
- Streamlit logs: Check the console output
- Application logs: Check `pipeline.log`
- Database logs: Check SQLite database file

## Support

For issues and questions:
1. Check the README.md for basic usage
2. Review the test files for examples
3. Open an issue on GitHub 