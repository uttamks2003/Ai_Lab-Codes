
# Experimnet 2 A(2)
#  program to randomly generate k-SAT problems

import random

def generate_k_sat(k, m, n):
    clauses = []
    variables = list(range(1, n+1))
    6
    for _ in range(m):
    clause = set()
    while len(clause) < k:
    variable = random.randint(1, n)
    negated = random.choice([True, False])
    if negated:
    variable = -variable
    clause.add(variable)
    clauses.append(clause)
    return clauses
    def print_k_sat(clauses):
    for i, clause in enumerate(clauses):
    clause_str = " v ".join(map(str, clause))
    print("(", clause_str, ")", end="")
    if i < len(clauses) - 1:
    print(" ∧ ", end="")
    print("\n")
    if __name__ == "__main__":
    k = int(input("Enter k (length of each clause): "))
    m = int(input("Enter m (number of clauses): "))
    n = int(input("Enter n (number of variables): "))
    7
    k_sat_problem = generate_k_sat(k, m, n)
    print("\nGenerated k-SAT problem:")
    print_k_sat(k_sat_problem)

# B
# C++ program to randomly generate k-SAT problems


include <bits/stdc++.h>
using namespace std;

vector< string > combs;

void getCombination(vector<string> arr, int n, int r, int index, vector<string> data, int i) {
    if (index == r)
    {
        string c = "";
        for (int j = 0; j < r; j++)
            c += data[j] + "^";
        combs.push_back(c);
        return;
    }
    if (i >= n)
        return;

    data[index] = arr[i];
    getCombination(arr, n, r, index + 1, data, i + 1);
    getCombination(arr, n, r, index, data, i+1);
}

void generateProblems(int k, int m, int n) {
    vector<string> posVars;
    char a = 'a';
    for(int i=0; i<n; i++) {
        string s;
        s.insert(0, 1, a);
        posVars.push_back(s);
        a++;
    }

    vector<string> negVars;
    for(int i=0; i<n; i++){
        negVars.push_back("~" + posVars[i]);
    }

    vector<string> totalVars = posVars;
    totalVars.insert(totalVars.begin()+n, negVars.begin(), negVars.end());

    vector<string> data(k);

    getCombination(totalVars, n*2, k, 0, data, 0);

    map<string, int> problems;
    string problem;

    for(int i=0; i<m; ) {
        srand(time(0));
        int pos = rand() % (n*2);
        string p = combs[pos];
        if (problems[p] == 0 ){
            i++;
            problems[p]++;
        }
    }

    for(auto i=problems.begin(); i!=problems.end(); i++){
        cout<<" ("<<(*i).first<<") ";
    }
    cout<<endl;
}

int main() {
    ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0);
    int k, m, n;
    cout<<"Enter length of each clause:\t"<<flush;
    cin >> k;
    cout<<"Enter number of clauses:\t"<<flush;
    cin >> m;
    cout<<"Enter number of variables:\t"<<flush;
    cin >> n;

    generateProblems(k, m, n);
    return 0;
}