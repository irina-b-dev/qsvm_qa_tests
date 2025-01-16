# qsvm_qa_tests

## Usage 



### **1. Create a New Virtual Environment**
Create a new Python virtual environment using the `venv` module or `virtualenv`.

#### Using `venv` (built-in):
```bash
python3 -m venv new-venv
```

#### Using `virtualenv` (if you prefer):
```bash
pip install virtualenv  # Install virtualenv if not already installed
virtualenv new-venv
```

---

### **2. Activate the Virtual Environment**
Activate the newly created virtual environment.

- On **Linux/macOS**:
  ```bash
  source new-venv/bin/activate
  ```

- On **Windows (Command Prompt)**:
  ```cmd
  new-venv\Scripts\activate
  ```

- On **Windows (PowerShell)**:
  ```powershell
  .\new-venv\Scripts\Activate.ps1
  ```


---

### **3. Install Packages from `requirements.txt`**
```bash
pip install -r requirements.txt
```

---

### **4. Verify Installation**
You can verify that the packages are installed in the virtual environment by listing them:
```bash
pip list
```


------
## With Docker


For Linux
- run `build_docker.sh`
- run `run_docker.sh`
