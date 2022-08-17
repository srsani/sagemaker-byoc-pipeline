tag_name=$1
# Name of algo -> ECR
algorithm_name=sagemaker-processing-redj

cd src/data_processing


account=$(aws sts get-caller-identity --query Account --output text)

region="us-east-1"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${tag_name}"
    
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}