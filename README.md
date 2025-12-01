# AI Rap Generator

A modular AI system that collaborates through a chat interface to create, evaluate, and refine rap lyrics using multiple cooperative agents.

---

## Problem

Writing rap lyrics is challenging because it requires:
- Creativity and theme consistency  
- Strong rhyme schemes and rhythmic structure  
- Iterative revision and feedback  
- Understanding of rap style and flow

Many creators struggle with generating polished lyrics efficiently.

---

## Solution

The AI Rap Generator uses a **multi-agent workflow** that automates:
- Idea planning and content generation
- Rhyme and style evaluation
- Structured revision and quality improvement
- Saving final results into files

This turns rap creation into a guided, iterative experience — suitable for both beginners and experienced writers.

---

## How the System Works

The system uses **specialized agents** that each play a role in the writing process:

| Agent | Responsibilities |
|-------|-----------------|
| RapChatManager | Coordinates user interaction and controls the workflow loop |
| RapWriter | Generates rap lyrics based on style and theme requests |
| RapEvaluator | Scores writing quality: rhyme, flow, cohesion, originality |
| RapSaver | Saves final lyrics to files for use and sharing |

Together, they run a **closed feedback loop** until quality goals are met.

---

## Architecture Overview

The workflow follows five main stages:

1. User provides theme, style, or inspiration
2. RapWriter generates a draft verse
3. RapEvaluator scores structure, rhyme, and clarity
4. ChatManager decides whether to revise or finalize
5. RapSaver exports finished lyrics

---

