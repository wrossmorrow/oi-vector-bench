
ssh_key_lambda := "~/.ssh/imagen-lambda.pem"
ssh_usr_lambda := "ubuntu"
ssh_key_aws := "~/.ssh/qwen-38-27b"
ssh_usr_aws := "ec2-user"

default:
    just --list

format: 
    uv run ruff format .
    uv run ruff check --fix .

check:
    uv run ruff format --check .

types:
    uv run mypy src

lambda-ssh ip user=ssh_usr_lambda ssh_key=ssh_key_lambda:
    just ssh {{ip}} {{user}} {{ssh_key}}

lambda-scp ip user=ssh_usr_lambda ssh_key=ssh_key_lambda:
    scp -i {{ssh_key}} -r .python-version README.md pyproject.toml uv.lock .env src scripts {{user}}@{{ip}}:~/

aws-scp ip user=ssh_usr_aws ssh_key=ssh_key_aws:
    scp -i {{ssh_key}} -r postgres {{user}}@{{ip}}:~/
    scp -i {{ssh_key}} -r .python-version README.md pyproject.toml uv.lock .env src scripts {{user}}@{{ip}}:~/
    scp -i {{ssh_key}} sweep.example.toml {{user}}@{{ip}}:~/

aws-ssh ip user=ssh_usr_aws ssh_key=ssh_key_aws:
    just ssh {{ip}} {{user}} {{ssh_key}}

ssh ip user ssh_key:
    ssh -i {{ssh_key}} {{user}}@{{ip}}
