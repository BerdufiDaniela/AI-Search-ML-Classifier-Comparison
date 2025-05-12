# AI_Search_And_ML_Classifier_Comparison
This repository contains the implementation of a university project in Artificial Intelligence, divided into two distinct parts:

# Part A – Bridge Crossing Problem
Implementation of the A* heuristic search algorithm to solve the well-known Bridge Crossing problem. The goal is to find an optimal strategy for a family to cross a bridge under specific time constraints.

# Problem Description
A group of family members must cross a bridge at night. The following constraints apply:

- The bridge can support a maximum of 2 people at a time.

- A torch is required to cross and must be brought back if anyone remains on the starting side.

- Each member has a different crossing time.

- When two people cross together, the time taken is equal to the slower person's time.

 #  Method
- State representation using the State class.

- Heuristic computation based on crossing times.

- Cost-based search using the formula f = g + h.

- The goal state is when all members have crossed to the other side.
  
# Code Structure
- FamMember.java – Class representing a family member.

- State.java – Class representing a state in the search space.

- mainClass.java – Executable program with user input support.

#  Technologies
- Language: Java

- Input via terminal

- Uses data structures like ArrayList and HashMap

 # Execution

javac mainClass.java
java mainClass
