FROM python:3.8

COPY requirements.txt .
RUN pip3 install -r requirements.txt 

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]

# FROM ubuntu:20.04
# # FROM docker

# RUN apt-get -y update && apt-get install -y --no-install-recommends \
#     wget \
#     python3 \
#     python3-pip \
#     nginx \
#     ca-certificates \
#     curl \ 
#     gnupg \
#     lsb-release 

# RUN apt install snapd -y

# # && rm -rf /var/lib/apt/lists/*
# RUN apt update
# RUN apt install -y apt-transport-https ca-certificates curl software-properties-common -y
# # RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add 
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# RUN echo \
#     "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
#     $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
# RUN apt-get update -y
# RUN apt-get install docker-ce docker-ce-cli containerd.io -y

# # RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
# # RUN apt-cache policy docker-ce
# # RUN apt-get install docker-ce docker-ce-cli containerd.io -y
# # RUN snap install docker

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt 

# # ENV HOME .
# # WORKDIR .
# CMD ["bash"]
