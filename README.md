
# Chat With PDF
This innovative tool allows users to effortlessly upload PDF documents and engage in interactive conversations about their content. Simply upload your PDF, and ask questions to receive instant answers, summaries, and insights.




## Demo
![ezgif-5-ff3cd61f9e](https://github.com/user-attachments/assets/a15a9fd7-0ecd-45ff-9cb5-e0536c66d815)



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

GOOGLE_API_KEY



## Setup the project locally

To deploy this project run
Navigate to the backend directory:
```bash
cd ../backend
```
Run the backend
1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
2. Load the requirements
```bash
pip install -r requirements.txt
```
3. Start the backend server
```bash
uvicorn main:app --reload
```

Navigate to the frontend directory:
```bash
cd ../frontend
```
Run the frontend
1. Install the dependencies
```bash
npm install
```
2. Start the frontend server
```bash
npm run dev
```

##  Project Structure

```sh
└── ChatWithPDF/
    ├── backend
    │   ├── .env
    │   ├── .gitignore
    │   ├── __pycache__
    │   ├── documents.db
    │   ├── main.py
    │   └── requirements.txt
    └── frontend
        ├── .gitignore
        ├── README.md
        ├── eslint.config.js
        ├── index.html
        ├── package-lock.json
        ├── package.json
        ├── postcss.config.js
        ├── public
        ├── src
        ├── tailwind.config.js
        └── vite.config.js
```
## Screenshots

![2](https://github.com/user-attachments/assets/fe9fdfc6-196b-447d-8947-1a5638e52a43)
![1](https://github.com/user-attachments/assets/33afe020-e6b1-4e9a-b252-2c7968adeb30)

