FROM public.ecr.aws/lambda/python:3.10

COPY requirements_bedrock.txt ./
# RUN python3.8 -m pip install -r requirements.txt -t .
RUN pip install -r requirements_bedrock.txt

COPY bedrock_app.py ./
COPY pipeline_context.py /var/lang/lib/python3.10/site-packages/pandasai/pipelines/
CMD ["bedrock_app.lambda_handler"]
