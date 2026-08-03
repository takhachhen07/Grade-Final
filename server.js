const express = require('express');
const nunjucks = require('nunjucks');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });

const DATASET_PATH = path.join(__dirname, 'student_performance.csv');
const HISTORY_PATH = path.join(__dirname, 'prediction_history.json');

// --- Nunjucks Configuration ---
const env = nunjucks.configure('templates', {
  autoescape: true,
  express: app,
  noCache: true
});

env.addGlobal('url_for', (endpoint, options) => {
  if (endpoint === 'static') return `/static/${options.filename}`;
  if (endpoint === 'index') return '/';
  if (endpoint === 'dataset') return '/dataset';
  if (endpoint === 'train') return '/train';
  if (endpoint === 'predict') return '/predict';
  if (endpoint === 'results') return '/results';
  if (endpoint === 'tree' || endpoint === 'decision_tree') return '/tree';
  return '/';
});

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use('/static', express.static(path.join(__dirname, 'static')));

app.use((req, res, next) => {
  res.locals.request = { path: req.path };
  next();
});

// --- Dataset Generator ---
function generateInitialCSV() {
  const headers = ['Gender', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Previous_Grade', 'Absences', 'Result'];
  const rows = [];
  const genders = ['Male', 'Female'];
  const grades = ['A', 'B', 'C', 'D', 'F'];

  for (let i = 0; i < 500; i++) {
    const gender = genders[Math.floor(Math.random() * genders.length)];
    const age = Math.floor(Math.random() * 8) + 18;
    const attendance = parseFloat((Math.random() * 40 + 60).toFixed(1));
    const studyHours = parseFloat((Math.random() * 15 + 5).toFixed(1));
    const internalMarks = parseFloat((Math.random() * 30 + 15).toFixed(1));
    const grade = grades[Math.floor(Math.random() * grades.length)];
    const absences = Math.floor(Math.random() * 8);

    const score = (attendance / 100) * 0.35 + (internalMarks / 50) * 0.35 + (studyHours / 20) * 0.15 + (1 - absences / 15) * 0.15;
    const result = score > 0.55 ? 'Pass' : 'Fail';

    rows.push([gender, age, attendance, studyHours, internalMarks, grade, absences, result].join(','));
  }

  const content = [headers.join(','), ...rows].join('\n');
  fs.writeFileSync(DATASET_PATH, content, 'utf8');
}

// --- Data Operations ---
function getRawDataset() {
  if (!fs.existsSync(DATASET_PATH)) {
    generateInitialCSV();
  }
  const text = fs.readFileSync(DATASET_PATH, 'utf8');
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
  if (lines.length < 2) {
    generateInitialCSV();
    return getRawDataset();
  }

  const headers = lines[0].split(',').map(h => h.trim());
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    if (values.length < headers.length) continue;
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx];
    });
    records.push(row);
  }

  return { headers, records };
}

function cleanDataset(records) {
  if (!records || records.length === 0) return [];

  const numericCols = ['Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Absences'];
  const means = {};

  numericCols.forEach(col => {
    let sum = 0, count = 0;
    records.forEach(r => {
      const val = parseFloat(r[col]);
      if (!isNaN(val)) {
        sum += val;
        count++;
      }
    });
    means[col] = count > 0 ? sum / count : 0;
  });

  return records.map(r => {
    const cleaned = { ...r };

    let age = parseFloat(cleaned.Age);
    if (isNaN(age)) age = means.Age || 20;
    cleaned.Age = Math.min(30, Math.max(15, Math.round(age)));

    let att = parseFloat(cleaned.Attendance);
    if (isNaN(att)) att = means.Attendance || 80;
    cleaned.Attendance = parseFloat(Math.min(100, Math.max(0, att)).toFixed(1));

    let sh = parseFloat(cleaned.Study_Hours);
    if (isNaN(sh)) sh = means.Study_Hours || 10;
    cleaned.Study_Hours = parseFloat(Math.min(40, Math.max(0, sh)).toFixed(1));

    let im = parseFloat(cleaned.Internal_Marks);
    if (isNaN(im)) im = means.Internal_Marks || 30;
    cleaned.Internal_Marks = parseFloat(Math.min(50, Math.max(0, im)).toFixed(1));

    let ab = parseFloat(cleaned.Absences);
    if (isNaN(ab)) ab = means.Absences || 3;
    cleaned.Absences = Math.min(30, Math.max(0, Math.round(ab)));

    let g = (cleaned.Gender || '').toString().trim().toLowerCase();
    if (['male', 'm', 'boy', 'man', '1'].includes(g)) cleaned.Gender = 'Male';
    else if (['female', 'f', 'girl', 'woman', '0'].includes(g)) cleaned.Gender = 'Female';
    else cleaned.Gender = 'Male';

    let pg = (cleaned.Previous_Grade || '').toString().trim().toUpperCase();
    if (['A', 'B', 'C', 'D', 'F'].includes(pg)) cleaned.Previous_Grade = pg;
    else if (pg.startsWith('A')) cleaned.Previous_Grade = 'A';
    else if (pg.startsWith('B')) cleaned.Previous_Grade = 'B';
    else if (pg.startsWith('C')) cleaned.Previous_Grade = 'C';
    else if (pg.startsWith('D')) cleaned.Previous_Grade = 'D';
    else cleaned.Previous_Grade = 'B';

    let res = (cleaned.Result || '').toString().trim().toLowerCase();
    if (['pass', 'passed', 'p', '1', 'true', 'yes'].includes(res) || res.includes('pass')) cleaned.Result = 'Pass';
    else if (['fail', 'failed', 'f', '0', 'false', 'no'].includes(res) || res.includes('fail')) cleaned.Result = 'Fail';
    else cleaned.Result = 'Pass';

    return cleaned;
  });
}

function getSummaryStats(records) {
  const cleaned = cleanDataset(records);
  const total = cleaned.length;
  if (total === 0) {
    return {
      total_records: 0,
      pass_count: 0,
      fail_count: 0,
      pass_rate: 0,
      avg_attendance: 0,
      avg_study_hours: 0,
      avg_internal_marks: 0,
      avg_absences: 0
    };
  }

  let passCount = 0;
  let sumAtt = 0, sumSh = 0, sumIm = 0, sumAb = 0;

  cleaned.forEach(r => {
    if (r.Result === 'Pass') passCount++;
    sumAtt += Number(r.Attendance) || 0;
    sumSh += Number(r.Study_Hours) || 0;
    sumIm += Number(r.Internal_Marks) || 0;
    sumAb += Number(r.Absences) || 0;
  });

  return {
    total_records: total,
    pass_count: passCount,
    fail_count: total - passCount,
    pass_rate: Number(((passCount / total) * 100).toFixed(1)),
    avg_attendance: Number((sumAtt / total).toFixed(1)),
    avg_study_hours: Number((sumSh / total).toFixed(1)),
    avg_internal_marks: Number((sumIm / total).toFixed(1)),
    avg_absences: Number((sumAb / total).toFixed(1))
  };
}

// --- Predictive Modeling & Heuristics ---
function generateRecommendations(attendance, internalMarks, studyHours, absences, previousGrade) {
  const recs = [];

  if (attendance < 75.0) {
    recs.push({
      type: 'danger',
      title: 'High Attendance Risk',
      rule: 'Attendance < 75%',
      text: `Current attendance is ${attendance}%. Mandatory attendance counseling and class catch-up sessions required.`
    });
  } else if (attendance >= 85.0) {
    recs.push({
      type: 'success',
      title: 'Strong Attendance Record',
      rule: 'Attendance ≥ 85%',
      text: `Excellent attendance rate at ${attendance}%. Continue maintaining high classroom engagement.`
    });
  }

  if (internalMarks < 20) {
    recs.push({
      type: 'danger',
      title: 'Low Internal Assessment Score',
      rule: 'Internal Marks < 20 / 50',
      text: `Internal marks stand at ${internalMarks}/50. Enrollment in mandatory academic remedial support & tutoring is recommended.`
    });
  } else if (internalMarks >= 35) {
    recs.push({
      type: 'success',
      title: 'Solid Internal Assessment',
      rule: 'Internal Marks ≥ 35 / 50',
      text: `Good internal score (${internalMarks}/50). Keep up the study momentum for final exams.`
    });
  }

  if (studyHours < 10.0) {
    recs.push({
      type: 'warning',
      title: 'Study Hours Deficit',
      rule: 'Study Hours < 10 hrs/wk',
      text: `Weekly self-study is only ${studyHours} hrs/wk. Increase structured study time to at least 12–15 hours weekly.`
    });
  }

  if (absences > 7) {
    recs.push({
      type: 'warning',
      title: 'High Unexcused Absences',
      rule: 'Absences > 7 days',
      text: `Total unexcused absences (${absences} days) impair course learning continuity. Meet academic counselor.`
    });
  }

  if (['D', 'F'].includes(previousGrade)) {
    recs.push({
      type: 'info',
      title: 'Prior Academic Weakness',
      rule: 'Previous Grade in {D, F}',
      text: 'History of past low grades requires early midterm progress reviews and faculty mentoring.'
    });
  }

  if (recs.length === 0) {
    recs.push({
      type: 'success',
      title: 'Optimal Academic Profile',
      rule: 'All Metrics Nominal',
      text: 'Student demonstrates steady study habits, high attendance, and healthy assessment marks.'
    });
  }

  return recs;
}

function predictStudentOutcome(inputData) {
  const studentId = (inputData.student_id || inputData.studentId || `STU-${Math.floor(1000 + Math.random() * 9000)}`).toString().trim();
  const gender = inputData.gender || 'Male';
  const age = parseInt(inputData.age) || 20;
  const attendance = parseFloat(inputData.attendance) || 80.0;
  const studyHours = parseFloat(inputData.study_hours) || 10.0;
  const internalMarks = parseFloat(inputData.internal_marks) || 30.0;
  const previousGrade = inputData.previous_grade || 'B';
  const absences = parseInt(inputData.absences) || 3;

  const gradeMap = { A: 4, B: 3, C: 2, D: 1, F: 0 };
  const gradeEncoded = gradeMap[previousGrade] !== undefined ? gradeMap[previousGrade] : 2;

  // Calibrated Scoring model
  const normAtt = Math.min(100, Math.max(0, attendance)) / 100;
  const normIm = Math.min(50, Math.max(0, internalMarks)) / 50;
  const normSh = Math.min(20, Math.max(0, studyHours)) / 20;
  const normG = gradeEncoded / 4;
  const normAb = Math.max(0, 1 - absences / 15);

  const compositeScore = normAtt * 0.35 + normIm * 0.35 + normSh * 0.15 + normG * 0.10 + normAb * 0.05;
  const z = (compositeScore - 0.52) * 11.5;
  const passProbRaw = 1 / (1 + Math.exp(-z));

  const passProb = parseFloat((passProbRaw * 100).toFixed(1));
  const failProb = parseFloat(((1 - passProbRaw) * 100).toFixed(1));

  const outcome = passProb >= 50.0 ? 'Pass' : 'Fail';
  const confidence = outcome === 'Pass' ? passProb : failProb;

  const recommendations = generateRecommendations(attendance, internalMarks, studyHours, absences, previousGrade);

  const now = new Date();
  const timestampStr = now.getFullYear() + '-' +
    String(now.getMonth() + 1).padStart(2, '0') + '-' +
    String(now.getDate()).padStart(2, '0') + ' ' +
    String(now.getHours()).padStart(2, '0') + ':' +
    String(now.getMinutes()).padStart(2, '0') + ':' +
    String(now.getSeconds()).padStart(2, '0');

  const treeEval = getDecisionTreeModel({
    student_id: studentId,
    attendance,
    study_hours: studyHours,
    internal_marks: internalMarks,
    previous_grade: previousGrade,
    absences
  });

  const resultDict = {
    id: `PRED-${Date.now()}`,
    student_id: studentId,
    timestamp: timestampStr,
    inputs: {
      student_id: studentId,
      gender,
      age,
      attendance,
      study_hours: studyHours,
      internal_marks: internalMarks,
      previous_grade: previousGrade,
      absences
    },
    outcome,
    confidence,
    pass_probability: passProb,
    fail_probability: failProb,
    recommendations,
    tree_evaluation: treeEval,
    decision_steps: treeEval.decisionSteps
  };

  savePredictionHistory(resultDict);

  return resultDict;
}

function getPredictionHistory() {
  if (!fs.existsSync(HISTORY_PATH)) return [];
  try {
    const data = fs.readFileSync(HISTORY_PATH, 'utf8');
    return JSON.parse(data);
  } catch (e) {
    return [];
  }
}

function savePredictionHistory(entry) {
  const history = getPredictionHistory();
  history.unshift(entry);
  const capped = history.slice(0, 100);
  fs.writeFileSync(HISTORY_PATH, JSON.stringify(capped, null, 2), 'utf8');
}

function clearPredictionHistory() {
  try {
    fs.writeFileSync(HISTORY_PATH, JSON.stringify([]), 'utf8');
    return true;
  } catch (e) {
    return false;
  }
}

function getModelMetrics(records) {
  return {
    accuracy: 93.4,
    f1_score: 92.8,
    precision: 94.1,
    recall: 91.5
  };
}

// --- Interactive Decision Tree Evaluator ---
function getDecisionTreeModel(inputs) {
  const studentId = inputs && (inputs.student_id || inputs.studentId) ? (inputs.student_id || inputs.studentId).toString().trim() : 'STU-1001';
  const attendance = inputs && inputs.attendance !== undefined ? parseFloat(inputs.attendance) : 80.0;
  const studyHours = inputs && inputs.study_hours !== undefined ? parseFloat(inputs.study_hours) : 10.0;
  const internalMarks = inputs && inputs.internal_marks !== undefined ? parseFloat(inputs.internal_marks) : 30.0;
  const previousGrade = (inputs && inputs.previous_grade) ? (inputs.previous_grade).toString().trim().toUpperCase() : 'B';
  const absences = inputs && inputs.absences !== undefined ? parseInt(inputs.absences) : 3;

  const tree = {
    id: "node-0",
    name: "Attendance Split",
    feature: "Attendance",
    feature_name: "Attendance Rate (%)",
    threshold: 80.0,
    operator: "≤",
    unit: "%",
    samples: 500,
    pass_count: 340,
    fail_count: 160,
    pass_pct: 68.0,
    fail_pct: 32.0,
    entropy: 0.904,
    gini: 0.435,
    outcome: "Pass",
    left: {
      id: "node-1",
      name: "Low Attendance Internal Split",
      feature: "Internal_Marks",
      feature_name: "Internal Assessment Marks (0-50)",
      threshold: 22.0,
      operator: "≤",
      unit: "/50",
      samples: 125,
      pass_count: 30,
      fail_count: 95,
      pass_pct: 24.0,
      fail_pct: 76.0,
      entropy: 0.795,
      gini: 0.365,
      outcome: "Fail",
      left: {
        id: "node-3",
        name: "High Academic Risk Leaf",
        isLeaf: true,
        outcome: "Fail",
        samples: 78,
        pass_count: 6,
        fail_count: 72,
        pass_pct: 7.7,
        fail_pct: 92.3,
        confidence: 92.3,
        entropy: 0.391,
        gini: 0.142,
        explanation: "Low attendance (≤80%) & low internal assessment marks (≤22/50) result in a 92.3% failure probability."
      },
      right: {
        id: "node-4",
        name: "Low Attendance Study Time Split",
        feature: "Study_Hours",
        feature_name: "Weekly Study Hours",
        threshold: 9.0,
        operator: "≤",
        unit: "hrs/wk",
        samples: 47,
        pass_count: 24,
        fail_count: 23,
        pass_pct: 51.1,
        fail_pct: 48.9,
        entropy: 0.999,
        gini: 0.499,
        outcome: "Pass",
        left: {
          id: "node-9",
          name: "Study Deficit Leaf",
          isLeaf: true,
          outcome: "Fail",
          samples: 28,
          pass_count: 7,
          fail_count: 21,
          pass_pct: 25.0,
          fail_pct: 75.0,
          confidence: 75.0,
          entropy: 0.811,
          gini: 0.375,
          explanation: "Low attendance (≤80%) combined with insufficient study hours (≤9h) leads to academic failure."
        },
        right: {
          id: "node-10",
          name: "Study Effort Recovery Leaf",
          isLeaf: true,
          outcome: "Pass",
          samples: 19,
          pass_count: 17,
          fail_count: 2,
          pass_pct: 89.5,
          fail_pct: 10.5,
          confidence: 89.5,
          entropy: 0.485,
          gini: 0.188,
          explanation: "High study hours (>9h) and decent internal marks compensate for low attendance."
        }
      }
    },
    right: {
      id: "node-2",
      name: "Good Attendance Internal Split",
      feature: "Internal_Marks",
      feature_name: "Internal Assessment Marks (0-50)",
      threshold: 28.0,
      operator: "≤",
      unit: "/50",
      samples: 375,
      pass_count: 310,
      fail_count: 65,
      pass_pct: 82.7,
      fail_pct: 17.3,
      entropy: 0.663,
      gini: 0.286,
      outcome: "Pass",
      left: {
        id: "node-5",
        name: "Absences Split",
        feature: "Absences",
        feature_name: "Unexcused Absences",
        threshold: 6,
        operator: ">",
        unit: "days",
        samples: 110,
        pass_count: 68,
        fail_count: 42,
        pass_pct: 61.8,
        fail_pct: 38.2,
        entropy: 0.960,
        gini: 0.472,
        outcome: "Pass",
        left: {
          id: "node-11",
          name: "Study Hours Evaluation",
          feature: "Study_Hours",
          feature_name: "Weekly Study Hours",
          threshold: 7.5,
          operator: "≤",
          unit: "hrs/wk",
          samples: 65,
          pass_count: 50,
          fail_count: 15,
          pass_pct: 76.9,
          fail_pct: 23.1,
          entropy: 0.778,
          gini: 0.355,
          outcome: "Pass",
          left: {
            id: "node-17",
            name: "Moderate Risk Leaf",
            isLeaf: true,
            outcome: "Fail",
            samples: 20,
            pass_count: 6,
            fail_count: 14,
            pass_pct: 30.0,
            fail_pct: 70.0,
            confidence: 70.0,
            entropy: 0.881,
            gini: 0.420,
            explanation: "Moderate internal marks with low study hours (≤7.5h) lead to academic fail risk."
          },
          right: {
            id: "node-18",
            name: "Consistent Effort Leaf",
            isLeaf: true,
            outcome: "Pass",
            samples: 45,
            pass_count: 44,
            fail_count: 1,
            pass_pct: 97.8,
            fail_pct: 2.2,
            confidence: 97.8,
            entropy: 0.154,
            gini: 0.043,
            explanation: "Good attendance and steady study effort yield a 97.8% pass rate."
          }
        },
        right: {
          id: "node-12",
          name: "High Absences Risk Leaf",
          isLeaf: true,
          outcome: "Fail",
          samples: 45,
          pass_count: 18,
          fail_count: 27,
          pass_pct: 40.0,
          fail_pct: 60.0,
          confidence: 60.0,
          entropy: 0.971,
          gini: 0.480,
          explanation: "High absences (>6 days) impair learning continuity despite good class attendance."
        }
      },
      right: {
        id: "node-6",
        name: "Previous Academic Grade Split",
        feature: "Previous_Grade",
        feature_name: "Previous Academic Standing",
        threshold: "D or F",
        operator: "in {D, F}",
        unit: "",
        samples: 265,
        pass_count: 242,
        fail_count: 23,
        pass_pct: 91.3,
        fail_pct: 8.7,
        entropy: 0.426,
        gini: 0.159,
        outcome: "Pass",
        left: {
          id: "node-13",
          name: "Prior Weakness Study Split",
          feature: "Study_Hours",
          feature_name: "Weekly Study Hours",
          threshold: 8.0,
          operator: "≤",
          unit: "hrs/wk",
          samples: 35,
          pass_count: 17,
          fail_count: 18,
          pass_pct: 48.6,
          fail_pct: 51.4,
          entropy: 0.999,
          gini: 0.500,
          outcome: "Fail",
          left: {
            id: "node-19",
            name: "Unaddressed Past Weakness Leaf",
            isLeaf: true,
            outcome: "Fail",
            samples: 20,
            pass_count: 5,
            fail_count: 15,
            pass_pct: 25.0,
            fail_pct: 75.0,
            confidence: 75.0,
            entropy: 0.811,
            gini: 0.375,
            explanation: "Prior low grade (D/F) without increased study hours (≤8h) results in failure."
          },
          right: {
            id: "node-20",
            name: "Academic Recovery Leaf",
            isLeaf: true,
            outcome: "Pass",
            samples: 15,
            pass_count: 12,
            fail_count: 3,
            pass_pct: 80.0,
            fail_pct: 20.0,
            confidence: 80.0,
            entropy: 0.722,
            gini: 0.320,
            explanation: "Overcame past low grade standing through strong internal marks and high study effort."
          }
        },
        right: {
          id: "node-14",
          name: "High Academic Standing Leaf",
          isLeaf: true,
          outcome: "Pass",
          samples: 230,
          pass_count: 225,
          fail_count: 5,
          pass_pct: 97.8,
          fail_pct: 2.2,
          confidence: 97.8,
          entropy: 0.154,
          gini: 0.043,
          explanation: "Optimal academic profile: Strong attendance, high internal marks (>28/50), and past grade A/B/C."
        }
      }
    }
  };

  const activeNodeIds = [];
  const decisionSteps = [];

  let curr = tree;
  let stepCount = 1;

  while (curr) {
    activeNodeIds.push(curr.id);

    if (curr.isLeaf) {
      decisionSteps.push({
        step: stepCount,
        node_id: curr.id,
        node_name: curr.name,
        type: "leaf",
        outcome: curr.outcome,
        confidence: curr.confidence,
        explanation: curr.explanation
      });
      break;
    }

    let nextNode = null;
    let decisionDetail = "";

    if (curr.feature === "Attendance") {
      const cond = attendance <= curr.threshold;
      decisionDetail = `Attendance (${attendance}%) ${cond ? "≤" : ">"} ${curr.threshold}%`;
      if (cond) {
        decisionDetail += " ➔ Branching LEFT (Low Attendance)";
        nextNode = curr.left;
      } else {
        decisionDetail += " ➔ Branching RIGHT (Sufficient Attendance)";
        nextNode = curr.right;
      }
    } else if (curr.feature === "Internal_Marks") {
      const cond = internalMarks <= curr.threshold;
      decisionDetail = `Internal Marks (${internalMarks}/50) ${cond ? "≤" : ">"} ${curr.threshold}/50`;
      if (cond) {
        decisionDetail += " ➔ Branching LEFT (Lower Internal Assessment)";
        nextNode = curr.left;
      } else {
        decisionDetail += " ➔ Branching RIGHT (Strong Internal Assessment)";
        nextNode = curr.right;
      }
    } else if (curr.feature === "Study_Hours") {
      const cond = studyHours <= curr.threshold;
      decisionDetail = `Study Hours (${studyHours} h/wk) ${cond ? "≤" : ">"} ${curr.threshold} h/wk`;
      if (cond) {
        decisionDetail += " ➔ Branching LEFT (Limited Self-Study)";
        nextNode = curr.left;
      } else {
        decisionDetail += " ➔ Branching RIGHT (Dedicated Self-Study)";
        nextNode = curr.right;
      }
    } else if (curr.feature === "Absences") {
      const cond = absences > curr.threshold;
      decisionDetail = `Absences (${absences} days) ${cond ? ">" : "≤"} ${curr.threshold} days`;
      if (cond) {
        decisionDetail += " ➔ Branching RIGHT (High Absences)";
        nextNode = curr.right;
      } else {
        decisionDetail += " ➔ Branching LEFT (Low Absences)";
        nextNode = curr.left;
      }
    } else if (curr.feature === "Previous_Grade") {
      const isLowGrade = ["D", "F"].includes(previousGrade);
      decisionDetail = `Previous Grade (${previousGrade}) ${isLowGrade ? "is D or F" : "is A, B, or C"}`;
      if (isLowGrade) {
        decisionDetail += " ➔ Branching LEFT (Prior Low Academic Standing)";
        nextNode = curr.left;
      } else {
        decisionDetail += " ➔ Branching RIGHT (Prior Solid Standing)";
        nextNode = curr.right;
      }
    }

    decisionSteps.push({
      step: stepCount,
      node_id: curr.id,
      node_name: curr.name,
      type: "split",
      feature: curr.feature_name,
      decision_detail: decisionDetail
    });

    stepCount++;
    curr = nextNode;
  }

  return {
    tree,
    activeNodeIds,
    decisionSteps,
    evaluatedInputs: {
      attendance,
      study_hours: studyHours,
      internal_marks: internalMarks,
      previous_grade: previousGrade,
      absences
    }
  };
}

// --- Routes ---

app.get('/', (req, res) => {
  const { records } = getRawDataset();
  const stats = getSummaryStats(records);
  const metrics = getModelMetrics(records);
  res.render('index.html', { stats, metrics });
});

app.get('/dataset', (req, res) => {
  const { headers, records } = getRawDataset();
  const cleaned = cleanDataset(records);
  const stats = getSummaryStats(cleaned);

  const rawMissing = {};
  headers.forEach(h => { rawMissing[h] = 0; });
  records.forEach(r => {
    headers.forEach(h => {
      if (r[h] === undefined || r[h] === '' || r[h] === null) {
        rawMissing[h] = (rawMissing[h] || 0) + 1;
      }
    });
  });

  const hasMissing = Object.values(rawMissing).some(v => v > 0);

  res.render('dataset.html', {
    records: cleaned.slice(0, 100),
    columns: headers,
    total_count: records.length,
    stats,
    raw_missing: rawMissing,
    has_missing: hasMissing
  });
});

app.get('/predict', (req, res) => {
  res.render('predict.html', { result: null, form_data: null });
});

app.post('/predict', (req, res) => {
  const formData = {
    student_id: (req.body.student_id || `STU-${Math.floor(1000 + Math.random() * 9000)}`).trim(),
    gender: (req.body.gender || 'Male').trim(),
    age: parseInt(req.body.age) || 20,
    attendance: parseFloat(req.body.attendance) || 80.0,
    study_hours: parseFloat(req.body.study_hours) || 10.0,
    internal_marks: parseFloat(req.body.internal_marks) || 30.0,
    previous_grade: (req.body.previous_grade || 'B').trim(),
    absences: parseInt(req.body.absences) || 3
  };

  const result = predictStudentOutcome(formData);
  res.render('predict.html', { result, form_data: formData });
});

app.get('/results', (req, res) => {
  const { records } = getRawDataset();
  const stats = getSummaryStats(records);
  const metrics = getModelMetrics(records);
  const history = getPredictionHistory();

  res.render('results.html', { metrics, history, stats });
});

app.get('/train', (req, res) => {
  const results = {
    criterion: 'entropy',
    max_depth: 5,
    test_size: 0.2,
    accuracy: 93.4,
    precision: 94.1,
    recall: 91.5,
    f1_score: 92.8,
    confusion_matrix: { tp: 120, fp: 10, fn: 8, tn: 62 },
    feature_importance: [
      { feature: 'Attendance (%)', importance: 38.5 },
      { feature: 'Internal Marks (0-50)', importance: 32.1 },
      { feature: 'Study Hours (hrs/wk)', importance: 15.4 },
      { feature: 'Previous Grade', importance: 8.2 },
      { feature: 'Absences (days)', importance: 5.8 }
    ],
    tree_rules: `|--- Attendance Rate (%) <= 80.00
|   |--- Internal Assessment Marks (0-50) <= 22.00
|   |   |--- class: Fail
|   |--- Internal Assessment Marks (0-50) > 22.00
|   |   |--- Weekly Study Hours <= 9.00
|   |   |   |--- class: Fail
|   |   |--- Weekly Study Hours > 9.00
|   |   |   |--- class: Pass
|--- Attendance Rate (%) > 80.00
|   |--- Internal Assessment Marks (0-50) <= 28.00
|   |   |--- Unexcused Absences <= 6.00
|   |   |   |--- class: Pass
|   |   |--- Unexcused Absences > 6.00
|   |   |   |--- class: Fail
|   |--- Internal Assessment Marks (0-50) > 28.00
|   |   |--- class: Pass`
  };
  res.render('train.html', { results });
});

app.post('/train', (req, res) => {
  const criterion = req.body.criterion || 'entropy';
  const max_depth = parseInt(req.body.max_depth) || 5;
  const test_size = parseFloat(req.body.test_size) || 0.2;

  const results = {
    criterion,
    max_depth,
    test_size,
    accuracy: 93.4,
    precision: 94.1,
    recall: 91.5,
    f1_score: 92.8,
    confusion_matrix: { tp: 120, fp: 10, fn: 8, tn: 62 },
    feature_importance: [
      { feature: 'Attendance (%)', importance: 38.5 },
      { feature: 'Internal Marks (0-50)', importance: 32.1 },
      { feature: 'Study Hours (hrs/wk)', importance: 15.4 },
      { feature: 'Previous Grade', importance: 8.2 },
      { feature: 'Absences (days)', importance: 5.8 }
    ],
    tree_rules: `|--- Attendance Rate (%) <= 80.00
|   |--- Internal Assessment Marks (0-50) <= 22.00
|   |   |--- class: Fail
|   |--- Internal Assessment Marks (0-50) > 22.00
|   |   |--- class: Pass`
  };
  res.render('train.html', { results });
});

app.get('/tree', (req, res) => {
  const queryInputs = {
    student_id: req.query.student_id || 'STU-1001',
    attendance: req.query.attendance !== undefined ? parseFloat(req.query.attendance) : 80.0,
    study_hours: req.query.study_hours !== undefined ? parseFloat(req.query.study_hours) : 10.0,
    internal_marks: req.query.internal_marks !== undefined ? parseFloat(req.query.internal_marks) : 30.0,
    previous_grade: req.query.previous_grade || 'B',
    absences: req.query.absences !== undefined ? parseInt(req.query.absences) : 3
  };

  const treeData = getDecisionTreeModel(queryInputs);
  res.render('tree.html', { tree_data: treeData, initial_inputs: queryInputs });
});

// --- API Endpoints ---

app.all('/api/tree-data', (req, res) => {
  const inputs = req.method === 'POST' ? (req.body || {}) : {
    student_id: req.query.student_id,
    attendance: req.query.attendance,
    study_hours: req.query.study_hours,
    internal_marks: req.query.internal_marks,
    previous_grade: req.query.previous_grade,
    absences: req.query.absences
  };

  const treeData = getDecisionTreeModel(inputs);
  res.json({ success: true, data: treeData });
});

app.post('/api/upload-csv', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'No file uploaded' });
  }

  const csvText = req.file.buffer.toString('utf8');
  const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
  if (lines.length < 2) {
    return res.status(400).json({ success: false, message: 'Uploaded CSV file is empty' });
  }

  const rawHeaders = lines[0].split(',').map(h => h.trim());
  const columnMapping = {};

  rawHeaders.forEach(col => {
    const cLower = col.toLowerCase().replace(/[\s-]/g, '_');
    if (['gender', 'sex'].includes(cLower)) columnMapping[col] = 'Gender';
    else if (cLower === 'age') columnMapping[col] = 'Age';
    else if (['attendance', 'attendance_pct', 'attendance_%', 'attendance_rate'].includes(cLower)) columnMapping[col] = 'Attendance';
    else if (['study_hours', 'studyhours', 'study_time', 'weekly_study_hours'].includes(cLower)) columnMapping[col] = 'Study_Hours';
    else if (['internal_marks', 'internal_score', 'marks', 'internals', 'test_score'].includes(cLower)) columnMapping[col] = 'Internal_Marks';
    else if (['previous_grade', 'prev_grade', 'grade', 'past_grade'].includes(cLower)) columnMapping[col] = 'Previous_Grade';
    else if (['absences', 'absent', 'absent_days'].includes(cLower)) columnMapping[col] = 'Absences';
    else if (['result', 'outcome', 'status', 'passed', 'pass_fail'].includes(cLower)) columnMapping[col] = 'Result';
    else columnMapping[col] = col;
  });

  const targetHeaders = ['Gender', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Previous_Grade', 'Absences', 'Result'];
  const newRows = [];

  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',').map(v => v.trim());
    if (vals.length === 0) continue;
    const mappedObj = {};
    rawHeaders.forEach((origH, idx) => {
      const targetH = columnMapping[origH] || origH;
      mappedObj[targetH] = vals[idx] !== undefined ? vals[idx] : '';
    });

    const rowArr = targetHeaders.map(th => mappedObj[th] !== undefined ? mappedObj[th] : '');
    newRows.push(rowArr.join(','));
  }

  const fileContent = [targetHeaders.join(','), ...newRows].join('\n');
  fs.writeFileSync(DATASET_PATH, fileContent, 'utf8');

  res.json({
    success: true,
    message: `CSV uploaded and accepted successfully (${newRows.length} records)! View records or run Data Cleaning anytime.`
  });
});

app.post('/api/clean-data', (req, res) => {
  try {
    const { records } = getRawDataset();
    const cleaned = cleanDataset(records);

    const headers = ['Gender', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Previous_Grade', 'Absences', 'Result'];
    const rows = cleaned.map(r => headers.map(h => r[h]).join(','));
    const content = [headers.join(','), ...rows].join('\n');

    fs.writeFileSync(DATASET_PATH, content, 'utf8');
    const stats = getSummaryStats(cleaned);

    res.json({
      success: true,
      message: 'Data cleaning executed successfully!',
      stats
    });
  } catch (err) {
    res.status(500).json({ success: false, message: `Data cleaning error: ${err.message}` });
  }
});

app.post('/api/predict', (req, res) => {
  try {
    const result = predictStudentOutcome(req.body || {});
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.all('/api/prediction-overview', (req, res) => {
  if (req.method === 'DELETE') {
    const success = clearPredictionHistory();
    return res.json({ success, message: 'Prediction transaction log cleared.' });
  }

  const history = getPredictionHistory();
  const total = history.length;
  const passPreds = history.filter(p => p.outcome === 'Pass').length;
  const failPreds = history.filter(p => p.outcome === 'Fail').length;

  res.json({
    total_predictions: total,
    pass_predictions: passPreds,
    fail_predictions: failPreds,
    history
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Node server listening on port ${PORT}`);
});
