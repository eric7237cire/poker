// MonteCarlo.cpp : Defines the entry point for the console application.
//

#include <Python.h>

#include <tchar.h>
#include "bool_array.h"
//#include "mersenne.h"
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
using namespace std;

// The handranks lookup table- loaded from HANDRANKS.DAT.
static int HR[32487834];

static bool load_datfile()
{
	printf("Loading HandRanks.DAT file...");
	memset(HR, 0, sizeof(HR));
	FILE * fin;
	fopen_s(&fin, "E:\\git\\poker\\XPokerEval\\XPokerEval.TwoPlusTwo\\HandRanks.dat", "rb");
	if (!fin)
		return false;
	size_t bytesread = fread(HR, sizeof(HR), 1, fin);	// get the HandRank Array
	fclose(fin);
	printf("complete.\n\n");

	return true;
}


// This function isn't currently used, but shows how you lookup
// a 7-card poker hand. pCards should be a pointer to an array
// of 7 integers each with value between 1 and 52 inclusive.
static int LookupHand(vector<int>& pCards)
{
	int p = HR[53 + pCards[0]];
	p = HR[p + pCards[1]];
	p = HR[p + pCards[2]];
	p = HR[p + pCards[3]];
	p = HR[p + pCards[4]];
	p = HR[p + pCards[5]];
	return HR[p + pCards[6]];
}



static void partial_fisher_yates(vector<uint32_t>& r, uint32_t k) {
	auto n = r.size();
	if (n < k) throw runtime_error("can't generate enough distinct elements in small interval");
		
	assert(n >= 1);
		
	for (uint32_t i = 0; i < n; ++i)
		r[i] = i;

	for (uint32_t i = 0; i < k; ++i) {		
		uint32_t z = i + rand() % (n-i);
		auto tmp = r[i];
		r[i] = r[z];
		r[z] = tmp;		
	}
	
}



static char StdDeck_rankChars[] = "23456789TJQKA";
static char StdDeck_suitChars[] = "hdcs";


static string cardToString(int cardIndex) {
	char r = StdDeck_rankChars[cardIndex / 4];
	char s_ch = StdDeck_suitChars[cardIndex % 4];
	
	string s;
	s.push_back(r);
	s.push_back(s_ch);
	return s;
}


//Returns 0 to 51
static int stringToCardIndex(const string& inString) {
	
	//cout << "Converting " << inString << " to an index" << endl;

	int rank, suit;

	for (rank = 0; rank <= 12; rank++)
		if (StdDeck_rankChars[rank] == toupper(inString[0]))
			break;
	if (rank > 12)
		return -1;
	
	for (suit = 0; suit <= 3; suit++)
		if (StdDeck_suitChars[suit] == tolower(inString[1]))
			break;
	if (suit > 3)
		return -1;
	
	//return suit * 13 + rank;
	return rank * 4 + suit;

}



static double run_simulation(int n_players_inc_hero, const vector<string>& hero_cards, vector<string> common_cards, int nSimulationsToDo, bool showOutput)
{

	//load_datfile();

	if (showOutput) printf("Testing the Two Plus Two 7-Card Evaluator\n");
	if (showOutput) printf("-----------------------------------------\n\n");

	// Load the HandRanks.DAT file and map it into the HR array
	if (hero_cards.size() != 2) {
		cerr << "Must provide heros cards" << endl;
		return -1;
	}

	if (common_cards.size() > 5) {
		cerr << "Too many common cards" << endl;
		return -1;
	}

	BoolArray taken(52, 0);

	int holeCard1 = stringToCardIndex(hero_cards[0]);
	int holeCard2 = stringToCardIndex(hero_cards[1]);
	taken.set(holeCard1);
	taken.set(holeCard2);

	if (holeCard1 == holeCard2) {
		cerr << "Cannot have same hole cards" << endl;
		return false;
	}

	vector<int> knownCommonCards;
	for (int i = 0; i < common_cards.size(); ++i) {
		auto idx = stringToCardIndex(common_cards[i]);
		knownCommonCards.push_back(idx);

		if (taken.get(idx)) {
			cerr << "Duplicate common cards" << endl;
			return false;
		}
		taken.set(idx);
	}

	vector<uint32_t> remaining_cards;
	for (int i = 0; i < 52; ++i)
	{
		if (taken.get(i) == false) {
			//store 1 - 52 since the hand evaluator wants them that way
			remaining_cards.push_back(i + 1);
		}
	}

	assert(50 - knownCommonCards.size() == remaining_cards.size());


	int cardsNeeded = (n_players_inc_hero - 1) * 2 + 5 - knownCommonCards.size();

	int wonCount = 0;
	double tiedCount = 0;

#if 0
	int distribution[13][52];

	for (int p = 0; p < 5; ++p)
	{
		for (int c = 0; c < 52; ++c)
		{
			distribution[p][c] = 0;
		}
	}
#endif

	vector<uint32_t> card_deck;
	card_deck.resize(remaining_cards.size());

	//init to n players by 7 cards (2 hole cards + 5 common cards (either know or to be simulated)
	vector<vector<int>> player_cards(n_players_inc_hero, vector<int>(7, 0));

	vector<int> scores(n_players_inc_hero, 0);

	//Adding 1 to make it 1-52
	player_cards[0][0] = holeCard1 + 1;
	player_cards[0][1] = holeCard2 + 1;

	for (int i = 0; i < nSimulationsToDo; ++i)
	{

		//LookupSingleHands();
		partial_fisher_yates(card_deck, cardsNeeded);


		for (int p = 0; p < n_players_inc_hero; ++p)
		{
			for (int c = 0; c<knownCommonCards.size(); ++c)
			{
				//Set common cards for everybody
				player_cards[p][2 + c] = knownCommonCards[c] + 1;
			}
			for (int c = 0; c < 5 - knownCommonCards.size(); ++c)
			{
				player_cards[p][2 + c + knownCommonCards.size()] = remaining_cards[card_deck[c]];
			}
		}
		int random_card_index = 5 - knownCommonCards.size();
		for (int p = 1; p < n_players_inc_hero; ++p)
		{
			player_cards[p][0] = remaining_cards[card_deck[random_card_index++]];
			player_cards[p][1] = remaining_cards[card_deck[random_card_index++]];
		}

#if 0
		for (int p = 0; p < nPlayers; ++p)
		{
			for (int c = 0; c<7; ++c)
			{
				if (player_cards[p][c] < 1 || player_cards[p][c] > 52) {
					throw "damn";
				}

				++distribution[p][player_cards[p][c] - 1];
			}
		}
#endif

		for (int p = 0; p < n_players_inc_hero; ++p)
		{
			scores[p] = LookupHand(player_cards[p]);
			//cout << "Score for " << p << " is " << scores[p] << ".  Category=" << (scores[p] >> 12) << endl;
			
			for (int c = 0; c < 7; ++c)
			{
				//cout << "Player " << p << " Card " << c << " is " << cardToString(player_cards[p][c]-1) << endl;
			}
		}

		int best_enemy_score = *std::max_element(scores.begin() + 1, scores.end());

		if (best_enemy_score < scores[0]) {
			++wonCount;
		}
		if (best_enemy_score == scores[0]) {
			//the tied equity depends on how many players tied
			int denom = 1;
			for (int j = 1; j < n_players_inc_hero; ++j) {
				if (scores[0] == scores[j]) {
					++denom;
				}
			}
			tiedCount += 1.0 / denom;
		}

	}

	//cout << "Did " << nSimulationsToDo << " Simulations" << endl;
	if (showOutput) cout << "Won: " << wonCount << " " << setprecision(4) << (100.0 * wonCount / nSimulationsToDo) << "%" << endl;
	//cout << "Tied: " << setprecision(1) << tiedCount << " " << setprecision(4) << (100.0 * tiedCount / nSimulationsToDo) << "%" << endl;

	double equity = (100.0 * (tiedCount + wonCount) / nSimulationsToDo);
	if (showOutput) cout << "Equity: " << setprecision(4) << equity << "%" << endl;

#if 0

	for (int p = 0; p < nPlayers; ++p)
	{
		for (int c = 0; c < 52; ++c)
		{
			cout << "Player " << p << " number of hits for card " << c << " is " << distribution[p][c] << endl;
		}
	}
#endif
	//cout << "Press any key" << endl;
	//std::cin.get();
	return equity;
}





static PyObject * 
test_args(PyObject *self, PyObject *args)
{
	const char *cards;
	int nPlayers;

	if (!PyArg_ParseTuple(args, "si", &cards, &nPlayers))
		return NULL;
	
	return PyLong_FromLong(42 + nPlayers);
}

static PyObject *
run_simulation_py(PyObject *self, PyObject *args)
{
	const char *hole_cards;
	const char *common_cards;
	int nPlayers;
	int nSimul;
	int showOutput;

	//int n_players_inc_hero, const vector<string>& hero_cards, vector<string> common_cards, int nSimulationsToDo)

	if (!PyArg_ParseTuple(args, "issip", &nPlayers, &hole_cards, &common_cards, &nSimul, &showOutput))
		return NULL;

	int l_hc = strlen(hole_cards);
	int l_cc = strlen(common_cards);

	if (l_hc != 4) {
		cerr << "Must pass exactly 4 characters as hole cards, like Js2h" << endl;
		return NULL;
	}

	vector<string> v_hole_cards;
	string hc1;
	hc1.push_back(hole_cards[0]);
	hc1.push_back(hole_cards[1]);
	string hc2;
	hc2.push_back(hole_cards[2]);
	hc2.push_back(hole_cards[3]);
	v_hole_cards.push_back(hc1);
	v_hole_cards.push_back(hc2);

	vector<string> v_common_cards;

	for (int c = 0; c < l_cc; c += 2)
	{
		string cc;
		cc.push_back(common_cards[c]);
		cc.push_back(common_cards[c+1]);
		v_common_cards.push_back(cc);
	}
	
	auto equity = run_simulation(nPlayers, v_hole_cards, v_common_cards, nSimul, showOutput);

	return PyFloat_FromDouble(equity);
	
}




static PyMethodDef PokerMethods[] = {
	
{"test_args",  test_args, METH_VARARGS,
"Testing arguments"},
{"run_simulation", run_simulation_py , METH_VARARGS, "Run a monte carlo simulation"},

{NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef pokermodule = {
	PyModuleDef_HEAD_INIT,
	"poker",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	PokerMethods
};

PyMODINIT_FUNC
PyInit_poker(void)
{
	PyObject *m;

	m = PyModule_Create(&pokermodule);
	if (m == NULL)
		return NULL;

	load_datfile();


	return m;
}

