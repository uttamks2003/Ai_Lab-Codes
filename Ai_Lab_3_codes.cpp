
# CPP Code

include <bits/stdc++.h>
using namespace std;

#define fast_io ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#define ll long long

unsigned long long iter = 100000;

class Graph{
    private:
        ll nodes;
        ll** a;
        vector<ll> tour;
        ll cost;
        long double T = 1000;

    public:
        Graph(ll n,ll** arr): nodes(n),a(arr)
        {}

        void formGraph(){
            for(int i=0;i<nodes;i++){
                for(int j=0;j<nodes;j++){
                    if(a[i][j]==-1){
                        if(i==j)
                            a[i][j]=0;
                        else
                            a[i][j] = rand()%100+1;
                            a[j][i] = a[i][j];
                    }
                }
            }
        }

        void getGraph(){
            for(ll i=0;i<nodes;i++){
                for(ll j=0;j<nodes;j++){
                    cout<<"| "<<a[i][j]<<"\t|";
                }
                cout<<"\n";
            }
        }

        long double getTemp(){
            return T;
        }

        //random starting tour:
        void tour_gen(){
            cost=0;
            tour.push_back(0);
            ll f[nodes] = {0};
            f[0] = 1;
            for(ll i=1;i<nodes;i++){
                ll temp;
                do{
                    temp = rand()%nodes;
                }
                while(f[temp]!=0);
                f[temp] = 1;
                cost+=a[*(tour.end()-1)][temp];
                tour.push_back(temp);
            }
        }

        void getTour(){
            for(int i=0;i<tour.size();i++){
                cout<<tour[i]<<" --> ";
            }
            cout<<0<<endl;
        }

        long double returnCost(){

            return cost;
        }


        ll getCost(vector<ll> t){
            ll c=0;
            for(int i=0;i<t.size()-1;i++){
                c+=a[t[i]][t[i+1]];
            }
            return c;
        }

        vector<ll> generateChild(vector<ll> t){
            vector<ll> temp;
                ll one,two;
                //calculate two random nodes:
                do{
                    one = rand()%nodes;
                    two = rand()%nodes;
                }
                while(one==two||one==0||two==0);

                temp = t;

                ll ex;
                ex = temp[one];
                temp[one] = temp[two];
                temp[two] = ex;

                return temp;
        }



        void anneling(){
            //cost = getCost(tour);
            //vector<ll> child_cost;
            // long double e =2.71812;

            while(iter>0){
                //generate child:
                vector<ll> temp = generateChild(tour);
                //calculate cost:

                //annealate:

                    //calculate probability:
                    ll newCost = getCost(temp);
                    ll diff = newCost - cost;
                    long double r = diff/T;
                    long double p = 1/(1+exp(-r));

                    if(diff<0){
                        //select
                        tour = temp;
                        cost = newCost;
                        T*=0.2;
                        //getTour();
                    }
                    else{
                        long double prob = 1 / rand();
                        if(p >= prob){
                            //select
                            tour = temp;
                            cost = newCost;
                            T*=0.2;
                        }
                    }
                    iter--;
            }
        }








};


int main() {
    srand(time(0));
    //fast_io;
    ll n;
    cout<<"enter the number of nodes:\n";
    cin>>n;
    ll** a = new ll*[n];
    ll ex[5] = {0};
    for(ll i=0;i<n;i++){
        a[i] = new ll[n];
        for(ll j=0;j<n;j++){
            a[i][j] = -1;
        }
    }

    Graph g(n,a);
    cout<<"Initial Temperature = "<<g.getTemp()<<endl;
    cout<<"Number of Iterations = "<<iter<<endl;

    g.formGraph();
    cout<<"Genrated graph is:\n";
    g.getGraph();
    cout<<"\n";
    g.tour_gen();
    cout<<"Initaial tour is:\n";
    g.getTour();
    cout<<"initial cost is:\n"<<g.returnCost()<<endl;
    g.anneling();
    cout<<"Final tour is:\n";
    g.getTour();
    cout<<"Final cost is:\n"<<g.returnCost()<<endl;
    //cout<<"\n"<<ttt<<endl;


}

#  TSP using genetic Algorithm

include <bits/stdc++.h>
using namespace std;

long int k = 10;        //Population Size
long int iter = 100000; //Generation Count
long int mutRate= 5;    //Mutation Rate

vector<long int> child1, child2;
map<long int, int> genes1, genes2;

// sort vector with pair
struct sort_pred
{
	bool operator()(const pair< vector<long int>, long double >& firstElem, const pair< vector<long int>, long double >& secondElem)
	{
		return firstElem.second < secondElem.second;
	}
};

//Graph of cities
class Graph{
    private:
        long int n;                             // No. of vertices
        vector< vector<long double > > adjList; // adjacency list
        vector<long int> tour;                  // storing tour
        long double cost;                       // cost of tour


    public:
        // Constructor
        Graph(long int n){
            this->n = n;
            adjList.resize(n, vector<long double>(n, -1));
            this->cost = 0;
        }

        long int returnNodes() {
            return n;
        }

        // function to add an edge to graph
        void addEdge(long int u, long int v, long double w)
        {
            adjList[u][v] = w;
            adjList[v][u] = w;
        }

        //initializing graph
        void formGraph(){
            for(long int i = 0; i < n; i++){
                for(long int j = 0; j < n; j++){
                    if(adjList[i][j] == -1){
                        adjList[i][j] = 0;
                    }
                }
            }
        }

        //allocating cost to each path
        void allocateCost(vector< pair<long int, long int> > coord){
            long int x = 0;
            for(auto i: coord) {
                long int y = 0;
                for(auto j: coord) {
                    long double d = sqrt(pow(i.first - j.first, 2) + pow(i.second - j.second, 2));
                    adjList[x][y] = d;
                    y++;
                }
                x++;
            }
        }

        //return cost of the tour
        long double returnCost(){
            return cost;
        }

        void printGraph(){
            for(long int i = 0; i < n; i++){
                for(long int j = 0; j < n; j++){
                    cout << "| " << adjList[i][j] << " |";
                }
                cout << endl;
            }
        }

        //generating a random tour
        void generateTour() {
            cost = 0;
            tour.push_back(0);

            vector<long double> freq(n);
            freq[0] = 1;
            for(long int i=1; i<n; i++) {
                long int temp;
                do{
                    temp = rand() % n;
                }
                while(freq[temp] != 0);
                freq[temp] = 1;
                cost += adjList[*(tour.end() - 1)][temp];
                tour.push_back(temp);
            }
        }

        vector<long int> getTour() {
            return tour;
        }

        void printTour(){
            for(long int i = 0; i<tour.size(); i++){
                cout << tour[i] << " --> ";
            }
            cout << 0 << endl;
        }

        //calculating cost of the tour
        long double calculateCost(vector<long int> t){
            long double c = 0;
            for(long int i=0; i<t.size()-1; i++){
                c += adjList[t[i]][t[i+1]];
            }
            return c;
        }

};


class GeneticAlgorithm {
    private:
        Graph *graph;                                               //graph of cities
        vector< pair< vector<long int>, long double> > population;  //to store tour and its cost
        long int populationSize;                                    //size of population
        long int iterationCount;                                    //generations count
        long int mutationRate;                                      //mutation rate
        long int realPopuSize;                                      //real population size

    public:
    //constructor
        GeneticAlgorithm(Graph *g, long int populationSize, long int iter, long int mutationRate) {
            this->populationSize = populationSize;
            this->graph = g;
            this->mutationRate = mutationRate;
            this->iterationCount = iter;
            this->realPopuSize = 0;
        }

        //to check if the parent chromosome exists in the population
        bool existsChromosome(vector<long int> &parent) {
            for(auto itr = population.begin(); itr != population.end(); ++itr) {
                vector<long int> &v = (*itr).first;
                if(equal(parent.begin(), parent.end(), v.begin())) {
                    return true;
                }
            }
            return false;
        }

        //initializing population
        void initPopulation( vector<long int> parent) {
            long double cost = graph->calculateCost(parent);
            if(isValid(parent)){
                population.push_back(make_pair(parent, cost));
                realPopuSize++;
            }

            long int n = graph->returnNodes();

            for (long int i=0; i<iterationCount; i++) {
                random_shuffle(parent.begin() + 1, parent.begin() + (rand() % (n - 1) + 1));
                long int cost = graph->calculateCost(parent);
                if(!existsChromosome(parent)) {
                    population.push_back(make_pair(parent, cost));
                    realPopuSize++;
                }
                if(realPopuSize == populationSize){
                    break;
                }
            }

            sort(population.begin(), population.end(), sort_pred());
        }

        void showPopulation() {
            for(auto itr = population.begin(); itr != population.end(); itr++) {
                vector<long int> &v = (*itr).first;
                for(int i=0; i<graph->returnNodes(); i++) {
                    cout << v[i] << " ";
                }
                cout << 0 << "\tCost: " << (*itr).second << endl;
            }

        }

        //crossover of two parents
        void crossOver(vector<long int>& parent1, vector<long int>& parent2) {
            child1.clear();
            child2.clear();
            genes1.clear();
            genes2.clear();
            for(int i=0; i<graph->returnNodes(); i++) {
                genes1[parent1[i]] = 0;
                genes2[parent2[i]] = 0;
            }
            int randP1 = rand() % (graph->returnNodes() - 1) + 1;
            int randP2 = rand() % (graph->returnNodes() - randP1) + randP1;
            if(randP1 == randP2) {
                if(randP1-1 > 1){
                    randP1--;
                }
                else if(randP2 + 1 < graph->returnNodes()) {
                    randP2++;
                }
                else {
                    long double prob = (rand() % 100) + 1;
                    if(prob > 50) {
                        randP2++;
                    }
                    else {
                        randP1--;
                    }
                }
            }
            for(long int i=0; i<randP1; i++) {
                child1.push_back(parent1[i]);
                child2.push_back(parent2[i]);
                genes1[parent1[i]] == 1;
                genes2[parent2[i]] == 1;
            }
            for(long int i=randP2+1; i<graph->returnNodes(); i++) {
                genes1[parent2[i]] = 1;
                genes2[parent2[i]] = 1;
            }
            for(long int i=randP2; i>= randP1; i--) {
                if(genes1[parent2[i]] == 0) {
                    child1.push_back(parent2[i]);
                    genes1[parent2[i]] = 1;
                }
                else {
                    for(auto itr=genes1.begin(); itr!=genes1.end(); ++itr) {
                        if(itr->second == 0) {
                            child1.push_back(itr->first);
                            genes1[itr->first] = 1;
                            break;
                        }
                    }
                }
                if(genes2[parent1[i]] == 0) {
                    child2.push_back(parent1[i]);
                    genes2[parent1[i]] = 1;
                }
                else {
                    for(auto itr=genes2.begin(); itr!=genes2.end(); ++itr) {
                        if(itr->second == 0) {
                            child2.push_back(itr->first);
                            genes2[itr->first] = 1;
                            break;
                        }
                    }
                }
            }
            for(long int i=randP2+1; i<graph->returnNodes(); i++) {
                child1.push_back(parent1[i]);
                child2.push_back(parent2[i]);
            }
            long int randMut = rand() % 100 + 1;
            if(randMut <= mutationRate) {
                long int a = rand() % (graph->returnNodes() - 1) + 1;
                long int b = rand() % (graph->returnNodes() - 1) + 1;

                int ax = child1[a];
                child1[a] = child1[b];
                child1[b] = ax;

                ax = child2[a];
                child2[a] = child2[b];
                child2[b] = ax;
            }
            long double cost1 = graph->calculateCost(child1);
            long double cost2 = graph->calculateCost(child2);
            if(!existsChromosome(child1) && isValid(child1))
                insertToPopulation(child1, cost1);
            if(!existsChromosome(child2) && isValid(child2))
                insertToPopulation(child2, cost2);
        }

        //to check wheather a solution is valid or not
        bool isValid(vector<long int>& sol) {
            set<long int> s;
            s.insert(sol.begin(), sol.end());
            if(s.size() != graph->returnNodes()){
                return false;
            }
            else{
                return true;
            }
        }

        //adding child to population
        void insertToPopulation(vector<long int>& child, long double cost) {
            long int l = 0;
            long int r = realPopuSize - 1;
            while(l <= r) {
                long int m = (l+r)/2;
                if(cost == population[m].second){
                    population.insert(population.begin()+m, make_pair(child, cost));
                    realPopuSize++;
                    return;
                }
                else if(cost > population[m].second) {
                    l = m + 1;
                }
                else {
                    r = m - 1;
                }
            }
            population.insert(population.begin()+l, make_pair(child, cost));
            realPopuSize++;
        }

        //Run Genetic Algorithm
        void run() {
            graph->generateTour();
            cout << "Initaial tour is:\n";
            graph->printTour();

            cout << "Initial cost is:\n" << graph->returnCost() << endl;
            vector<long int> parent = graph->getTour();
            initPopulation(parent);
            if(realPopuSize == 0) {
                return;
            }


            for (long int i=0; i<iterationCount; i++) {
                int psize = realPopuSize;
                if(realPopuSize >= 2) {
                    if(realPopuSize == 2) {
                        crossOver(population[0].first, population[1].first);
                    }
                    else {
                        long int parent1 = rand() % realPopuSize;
                        long int parent2 = rand() % realPopuSize;
                        while(parent1 == parent2) {
                            parent1 = rand() % realPopuSize;
                            parent2 = rand() % realPopuSize;
                        }
                        crossOver(population[parent1].first, population[parent2].first);
                    }

                    if ( realPopuSize > populationSize) {
                        long int d = realPopuSize - psize;
                        if(d == 1 || d == 2) {
                            population.pop_back();
                            if (d == 2)
                                population.pop_back();
                            realPopuSize -= d;
                        }
                    }

                }
                else {
                    crossOver(population[0].first, population[0].first);
                    if(realPopuSize > psize) {
                        population.pop_back();
                        realPopuSize--;
                    }
                }
            }

        }

        void showSolution() {
            for(long int i=0; i<population[0].first.size(); i++) {
                cout << population[0].first[i] << " --> ";
            }
            cout << 0 << endl;
            cout << "\nFinal Cost:\t" << population[0].second;
        }

        long double getCostOfSolution() {
            if(realPopuSize > 0) {
                return population[0].second;
            }
            return -1;
        }

};

int main() {
    //fast input and output
    ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0);

    //seeding random function with time
    srand(time(0));

    //number of cities
    long int n;
    cin >> n;

    long int optCost;
    cin >> optCost;

    vector< pair<long int, long int> > coord;
    for(long int i=0; i<n; i++) {
        long int num, x, y;
        cin >> num >> x >> y;
        coord.push_back(make_pair(x, y));
    }

    Graph * g = new Graph(n);
    g->formGraph();
    g->allocateCost(coord);


    GeneticAlgorithm ga(g, k, iter, mutRate);

    ga.run();
    ga.showPopulation();
    ga.showSolution();
    cout<<"\nOptimal Cost:\t"<<optCost<<endl;

    return 0;
}