import os
import sys
import json
import shutil
import subprocess
import ast
import re
import logging
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set your OpenAI API key
openai.api_key = config.get('openai_api_key')

class AIDevelopmentAssistant:
    def __init__(self):
        self.templates = {
            "flask": ["app.py", "requirements.txt", "static/", "templates/", "config.py"],
            "django": ["manage.py", "requirements.txt", "app/", "static/", "templates/", "settings.py"],
            "express": ["index.js", "package.json"],
            "react": ["src/", "public/", "package.json"],
            "vue": ["src/", "public/", "package.json"]
        }
        self.custom_templates_file = "custom_templates.json"
        self.boilerplate_file = "boilerplate.json"
        self.load_custom_templates()
        self.load_boilerplate()

    def load_custom_templates(self):
        if os.path.exists(self.custom_templates_file):
            with open(self.custom_templates_file, "r") as file:
                self.templates.update(json.load(file))
            logging.info("Custom templates loaded successfully.")
        else:
            logging.info("No custom templates found.")

    def load_boilerplate(self):
        if os.path.exists(self.boilerplate_file):
            with open(self.boilerplate_file, "r") as file:
                self.boilerplate = json.load(file)
            logging.info("Boilerplate content loaded successfully.")
        else:
            logging.info("No boilerplate content found.")
            self.boilerplate = {}

    def create_project_structure(self, project_name, template):
        if template not in self.templates:
            logging.error("Unsupported template!")
            print("Unsupported template!")
            return

        os.makedirs(project_name, exist_ok=True)
        for item in self.templates[template]:
            path = os.path.join(project_name, item)
            if item.endswith("/"):
                os.makedirs(path, exist_ok=True)
            else:
                with open(path, "w") as file:
                    file.write(self.boilerplate.get(item, ""))
        self.generate_readme(project_name, template)
        logging.info(f"{template.capitalize()} project '{project_name}' created successfully!")
        print(f"{template.capitalize()} project '{project_name}' created successfully!")

    def add_custom_template(self, template_name, files):
        if template_name in self.templates:
            logging.error("Template already exists!")
            print("Template already exists!")
            return

        self.templates[template_name] = files
        self.save_custom_templates()
        logging.info(f"Custom template '{template_name}' added successfully!")
        print(f"Custom template '{template_name}' added successfully!")

    def list_templates(self):
        logging.info("Listing available templates.")
        print("Available templates:")
        for template in self.templates:
            print(f"- {template}")

    def generate_readme(self, project_name, template):
        readme_content = f"# {project_name.capitalize()}\n\nThis is a {template.capitalize()} project.\n\n## Project Structure\n\n"
        for item in self.templates[template]:
            readme_content += f"- {item}\n"
        readme_path = os.path.join(project_name, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(readme_content)
        logging.info("README.md generated successfully!")
        print("README.md generated successfully!")

    def generate_server_config(self, server_type, domain):
        if server_type == "apache":
            config = f"""
<VirtualHost *:80>
    ServerName {domain}
    DocumentRoot /var/www/{domain}
    ErrorLog ${{APACHE_LOG_DIR}}/{domain}_error.log
    CustomLog ${{APACHE_LOG_DIR}}/{domain}_access.log combined
</VirtualHost>
"""
        elif server_type == "nginx":
            config = f"""
server {{
    listen 80;
    server_name {domain};
    root /var/www/{domain};
    index index.html;
}}"""
        else:
            logging.error("Unsupported server type!")
            print("Unsupported server type!")
            return

        with open(f"{server_type}_config.conf", "w") as f:
            f.write(config)
        logging.info(f"{server_type.capitalize()} configuration file created!")
        print(f"{server_type.capitalize()} configuration file created!")

    def lint_code(self, file_path):
        try:
            if file_path.endswith(".js"):
                output = subprocess.run(["eslint", file_path, "--fix"], capture_output=True, text=True)
            elif file_path.endswith(".php"):
                output = subprocess.run(["phpcs", "--standard=PSR12", file_path], capture_output=True, text=True)
            else:
                output = subprocess.run(["flake8", file_path], capture_output=True, text=True)
            logging.info(f"Linting completed for {file_path}.")
            print(output.stdout if output.stdout else "No issues found!")
        except FileNotFoundError as e:
            logging.error(f"Linting tool not found: {e}")
            print(f"Linting tool not found: {e}")

    def analyze_code(self, file_path):
        try:
            with open(file_path, "r") as file:
                code = file.read()
                tree = ast.parse(code)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                logging.info(f"Code analysis completed for {file_path}.")
                print(f"Found functions: {functions}")
        except Exception as e:
            logging.error(f"Error analyzing code: {e}")
            print(f"Error analyzing code: {e}")

    def predict_code_errors(self, code_snippet):
        X_test = self.vectorizer.transform([code_snippet])
        prediction = self.classifier.predict(X_test)
        logging.info("Code error prediction completed.")
        print(f"Predicted issue: {prediction[0]}")

    def suggest_code_optimizations(self, code_snippet):
        suggestions = []
        if "for " in code_snippet and "range" in code_snippet:
            suggestions.append("Consider using list comprehensions for better performance.")
        if "while True" in code_snippet:
            suggestions.append("Avoid infinite loops unless necessary.")
        logging.info("Code optimization suggestions generated.")
        print("Code optimization suggestions:" if suggestions else "No suggestions.")
        for suggestion in suggestions:
            print(f"- {suggestion}")

    def static_analysis(self, file_path):
        try:
            with open(file_path, "r") as file:
                code = file.read()
                issues = []
                if "eval(" in code:
                    issues.append("Use of eval() detected, which can lead to security vulnerabilities.")
                if "exec(" in code:
                    issues.append("Use of exec() detected, which can lead to security vulnerabilities.")
                logging.info(f"Static analysis completed for {file_path}.")
                print("Security issues found:" if issues else "No security issues found.")
                for issue in issues:
                    print(f"- {issue}")
        except Exception as e:
            logging.error(f"Error during static analysis: {e}")
            print(f"Error during static analysis: {e}")

    def deploy_docker(self, project_name):
        dockerfile_content = f"""
# Dockerfile for {project_name}
FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
"""
        with open(os.path.join(project_name, "Dockerfile"), "w") as dockerfile:
            dockerfile.write(dockerfile_content)
        logging.info("Dockerfile created successfully!")
        print("Dockerfile created successfully!")

    def deploy_kubernetes(self, project_name):
        k8s_deployment = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {project_name}-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {project_name}
  template:
    metadata:
      labels:
        app: {project_name}
    spec:
      containers:
      - name: {project_name}
        image: {project_name}:latest
        ports:
        - containerPort: 80
"""
        with open(os.path.join(project_name, "k8s_deployment.yaml"), "w") as k8s_file:
            k8s_file.write(k8s_deployment)
        logging.info("Kubernetes deployment file created successfully!")
        print("Kubernetes deployment file created successfully!")

    def setup_ci_cd(self, project_name):
        ci_cd_config = f"""
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest
"""
        os.makedirs(os.path.join(project_name, ".github", "workflows"), exist_ok=True)
        with open(os.path.join(project_name, ".github", "workflows", "ci-cd.yml"), "w") as ci_cd_file:
            ci_cd_file.write(ci_cd_config)
        logging.info("CI/CD pipeline configuration created successfully!")
        print("CI/CD pipeline configuration created successfully!")

    def integrate_github(self, repo_name):
        try:
            subprocess.run(["gh", "repo", "create", repo_name, "--public", "--source=.", "--remote=origin"], check=True)
            logging.info(f"GitHub repository '{repo_name}' created and linked successfully!")
            print(f"GitHub repository '{repo_name}' created and linked successfully!")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating GitHub repository: {e}")
            print(f"Error creating GitHub repository: {e}")

    def real_time_logging(self, log_file):
        try:
            with open(log_file, "r") as file:
                for line in file:
                    print(line, end="")
            logging.info(f"Real-time logging from {log_file}.")
        except FileNotFoundError:
            logging.error(f"Log file '{log_file}' not found.")
            print(f"Log file '{log_file}' not found.")

if __name__ == "__main__":
    assistant = AIDevelopmentAssistant()
    while True:
        print("\nAI Development Assistant:")
        print("1. Create Project Structure")
        print("2. Generate Server Config")
        print("3. Lint Code")
        print("4. Analyze Code")
        print("5. Predict Code Errors")
        print("6. Add Custom Template")
        print("7. List Templates")
        print("8. Suggest Code Optimizations")
        print("9. Perform Static Analysis")
        print("10. Deploy with Docker")
        print("11. Deploy with Kubernetes")
        print("12. Setup CI/CD Pipeline")
        print("13. Integrate with GitHub")
        print("14. Real-time Logging")
        print("15. Get Code Completion and Snippet Suggestions")
        print("16. Exit")
        choice = input("Select an option: ")

        if choice == "1":
            name = input("Enter project name: ")
            template = input("Enter template (flask/django/express/react/vue/laravel/spring_boot/angular/ruby_on_rails/asp_net/nodejs/symfony/codeigniter/svelte/nextjs/nuxtjs/fastapi/flask_restful/electron/flutter/django_rest_framework): ")
            assistant.create_project_structure(name, template)
        elif choice == "2":
            server = input("Enter server type (apache/nginx): ")
            domain = input("Enter domain name: ")
            assistant.generate_server_config(server, domain)
        elif choice == "3":
            path = input("Enter file path: ")
            assistant.lint_code(path)
        elif choice == "4":
            path = input("Enter file path: ")
            assistant.analyze_code(path)
        elif choice == "5":
            code = input("Enter code snippet: ")
            assistant.predict_code_errors(code)
        elif choice == "6":
            template_name = input("Enter custom template name: ")
            files = input("Enter files and directories (comma-separated): ").split(',')
            assistant.add_custom_template(template_name, files)
        elif choice == "7":
            assistant.list_templates()
        elif choice == "8":
            code = input("Enter code snippet: ")
            assistant.suggest_code_optimizations(code)
        elif choice == "9":
            path = input("Enter file path: ")
            assistant.static_analysis(path)
        elif choice == "10":
            name = input("Enter project name: ")
            assistant.deploy_docker(name)
        elif choice == "11":
            name = input("Enter project name: ")
            assistant.deploy_kubernetes(name)
        elif choice == "12":
            name = input("Enter project name: ")
            assistant.setup_ci_cd(name)
        elif choice == "13":
            repo = input("Enter GitHub repository name: ")
            assistant.integrate_github(repo)
        elif choice == "14":
            log_file = input("Enter log file path: ")
            assistant.real_time_logging(log_file)
        elif choice == "15":
            code_context = input("Enter code context: ")
            model_choice = input("Enter model choice (gpt-2/gpt-3/gpt-4): ").strip().lower()
            completions = assistant.get_code_completion(code_context, model_choice)
            print("Code Completions:")
            for i, completion in enumerate(completions, 1):
                print(f"{i}. {completion}")
        elif choice == "16":
            logging.info("Exiting AI Development Assistant.")
            sys.exit("Exiting AI Development Assistant.")
        else:
            logging.warning("Invalid choice. Try again!")
            print("Invalid choice. Try again!")