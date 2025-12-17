1. In this problem formulation, is there user/task priority or all are treated equally?  

Yes, there is a weight w(i,j) factor that indicates priority. In the toy example presented in the presentation, all weights were 1 for simplicity. In real life, you can adjust weight depending on the type of user, past usage, etc.

2. "People can game the system" - a clearer explanation is needed.

Through trial and error or by looking at how the system works from the inside (profiling), you can uncover many vulnerabilities and exploit these. Mentioned in depth in background section of the paper with examples from Google, Linux, Hadoop.

3. "Random allocation" - user requests 36 nodes and gets 29, not going to work for many problems.

Yes, random allocation is an example of perfect strategy-proofness. It's obviously not a viable solution because demands become completely irrelevant. We want to enforce a solution that still has consideration of demands such that the allocatin is close to what they asked for.

4. The optimization problem needs to explained better. 

Parallelization is the key. 

5. "Optimal job allocation table" - what is the gurantee a job can be split among multiple workers?

Job is not split amongst multiple workers. It just spends a portion of time at one worker (i.e. powerful CPU) before having to wait or move to another worker (i.e. weaker CPU).

6. How the modified DeDe algorithm solves the resource allocation problem is not very clear. At least more information is required to understand the feasibility of the proposed approach.

Mentioned in Paper.

7. Experimental evaluation is missing. What would be your comparison baseline?

We tested the result of adding noise to different stages in DeDe as well as different types of noise. RDP was the winner and adding noise at the column-update stage (demand update for each job) was ideal and beats adding noise before convergence, after convergence or during row-update stage (resource update for each resource).