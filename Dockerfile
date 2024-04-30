FROM python:3.11-slim
WORKDIR /app

RUN useradd -m bedrock && chown -R bedrock /app
USER bedrock
ENV PATH="/home/bedrock/.local/bin:${PATH}"

COPY --chown=bedrock:bedrock app.py settings.py cmn_auth.py .env /app/
COPY --chown=bedrock:bedrock requirements.txt /app/
COPY --chown=bedrock:bedrock pages /app/pages
COPY --chown=bedrock:bedrock .streamlit /app/.streamlit
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip3 install -r requirements.txt

EXPOSE 8501
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableWebsocketCompression=false"]
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--browser.gatherUsageStats=false"]

