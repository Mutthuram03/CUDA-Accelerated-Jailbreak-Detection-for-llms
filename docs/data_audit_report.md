# Dataset Quality & Security Audit Report

## 1. Overview & Integrity
- **Total Records Analyzed**: 39,234
- **SHA-256 Checksum**: `57f976448f05217cb07725f4bc3050145ab8461171359b09b8fa097b4e1b38b5`
- **Null / Missing Prompts**: 0
- **Empty Prompts**: 0
- **Invalid Labels**: 0
- **Exact Duplicate Prompts**: 0 (Unique: 39,234)

## 2. Class Distribution
| Class | Count | Percentage |
| :--- | :--- | :--- |
| **Benign (0)** | 19,617 | 50.00% |
| **Malicious / Adversarial (1)** | 19,617 | 50.00% |
| **Total** | 39,234 | 100.00% |

## 3. Prompt Length Distributions (Character Counts)
| Metric | Benign Prompts (0) | Malicious Prompts (1) |
| :--- | :--- | :--- |
| **Mean Length** | 62.9 chars | 503.1 chars |
| **Median (p50)** | 55.0 chars | 61.0 chars |
| **Min Length** | 1 chars | 10 chars |
| **Max Length** | 326 chars | 55089 chars |

## 4. Adversarial Attack Taxonomy Breakdown (Malicious Partition)
| Attack Signature Category | Detected Hits | % of Malicious Partition |
| :--- | :--- | :--- |
| **Instruction-Conflict Overrides** (e.g., `ignore previous`, `bypass above`) | 3,427 | 17.47% |
| **Persona Adoption & Jailbreak Framing** (e.g., `DAN`, `Developer Mode`) | 848 | 4.32% |
| **Base64 Payload Obfuscation** | 1,559 | 7.95% |
| **Contiguous Hex Encoding** | 60 | 0.31% |
| **Zero-Width Character Infiltration** | 20 | 0.10% |
