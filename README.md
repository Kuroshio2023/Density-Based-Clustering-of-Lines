
# Density-Based Spatial Clustering of Lines (DeLi)
A report of our work is available at https://arxiv.org/abs/2410.02290
This repository contains the official implementation of **DeLi**, a density-based clustering algorithm for **lines and line segments in high-dimensional spaces**, with applications to **incomplete data clustering**.  
The method generalizes DBSCAN to operate on geometric objects lacking a valid metric distance, using a **probabilistic neighbourhood generation framework**.

---

## 📘 Overview

Classical clustering methods focus on points and often fail for higher-order objects such as lines, especially when:
- No valid line–line metric satisfies triangle inequality,  
- Clusters have varying densities or non-convex geometry,  
- Data contain **missing entries**.

**DeLi** addresses these challenges by:
- Assigning a **probability density function (PDF)** along each line,  
- Generating a **custom neighbourhood of fixed volume**,  
- Using an **asymmetric neighbourhood relation** to form density-based clusters,  
- Naturally detecting **noise/outliers**,  
- Unifying clustering of **lines + points with missing values**.

---

## 🧩 Key Features
- Probabilistic neighbourhood generation via a continuous PDF \( f_l \)  
- Support for **three algorithmic versions** (metric-based, PDF-based, scaling-based)  
- Handles **arbitrary shapes**, **arbitrary cluster sizes**, and **noise**  
- Applicable to:
  - Synthetic and real-world line datasets (rail networks, subway maps)
  - High-dimensional point datasets with **one missing attribute**
- Time complexity: **O(n²)**  
- No need to pre-specify the number of clusters  

---

## 📂 Repository Structure
