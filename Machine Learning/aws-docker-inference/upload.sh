export AWS_PROFILE=uade-valorar
echo "\n*** Note: sam build needs docker server running in the host ***\n"  
sam build && sam deploy
