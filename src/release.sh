branch_name=$1

pip3 install -r .github/workflows/requirements.txt 

if [ $branch_name == "main" ];
then
    echo $branch_name
    python3 src/pipeline.py $branch_name
fi

if [ $branch_name == "staging" ];
then
    echo $branch_name
    python3 src/pipeline.py $branch_name

fi

if [ $branch_name == "production" ];
then
    echo $branch_name
    python3 src/pipeline.py $branch_name
fi

