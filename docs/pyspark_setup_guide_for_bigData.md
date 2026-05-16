# 🚀 Java + Python + Hadoop Setup on Windows (Quick Guide)

## ✅ Step 1: Install Java SDK (JDK 17)

Download JDK 17 from the [Adoptium Temurin Releases Page](https://adoptium.net/en-GB/temurin/releases?version=17&os=any&arch=any).

Install this exact version:
`OpenJDK17U-jdk_x64_windows_hotspot_17.0.19_10`

Default install path:
`C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot`

---

## ✅ Step 2: Install Python 3.11

Download Python 3.11 from the [Python 3.11.0 Download Page](https://www.python.org/downloads/release/python-3110/).

⚠️ **During installation, make sure to check:**

- [x] **Add Python to PATH**

Default install path:
`C:\Python311`

Verify the installation in your terminal:

```bash
python --version
```

**Expected output:**

```bash
Python 3.11.x
```

---

## ✅ Step 3: Install Hadoop Windows Files

Download the required binaries directly from GitHub:

- 🔹 [winutils.exe](https://github.com/cdarlint/winutils/blob/master/hadoop-3.3.6/bin/winutils.exe)
- 🔹 [hadoop.dll](https://github.com/cdarlint/winutils/blob/master/hadoop-3.3.6/bin/hadoop.dll)

---

## ✅ Step 4: Create Hadoop Folder

1. Create a new folder at this path: `C:\hadoop\bin`
2. Place both downloaded files inside that folder:
   - `C:\hadoop\bin\winutils.exe`
   - `C:\hadoop\bin\hadoop.dll`

---

## ✅ Step 5: Configure Environment Variables

Open your system configuration:
`System Properties → Advanced → Environment Variables`

### Add HADOOP_HOME

- **Variable Name:** `HADOOP_HOME`
- **Variable Value:** `C:\hadoop`
- **Add to PATH:** `%HADOOP_HOME%\bin`

---

### Add JAVA_HOME

- **Variable Name:** `JAVA_HOME`
- **Variable Value:** `C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot`
- **Add to PATH:** `%JAVA_HOME%\bin`

---

### Add PYTHON_HOME (Optional)

- **Variable Name:** `PYTHON_HOME`
- **Variable Value:** `C:\Python311`
- **Add to PATH:** `%PYTHON_HOME%`

---

## ✅ Step 6: Verify Installation

Open **Command Prompt (CMD)** and execute the following checks:

```cmd
java --version
```

```cmd
python --version
```

```cmd
where winutils
```

```cmd
echo %HADOOP_HOME%
```

---

## ✅ Final Structure

Your file directory should match this tree layout:

```text
C:\
 ├── hadoop
 │    └── bin
 │         ├── winutils.exe
 │         └── hadoop.dll
 │
 └── Python311
```

🎉 **Java 17 + Python 3.11 + Hadoop setup completed successfully on Windows.**
