include .env

aws_connect:
	@echo "Connecting to AWS instance..."
	ssh -i ${AWS_KEY} -D 5555 $(AWS_SSH_USER)@$(AWS_SSH_HOST)

	
