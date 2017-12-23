// MonteCarlo.cpp : Defines the entry point for the console application.
//

//#include <windows.h>
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
int HR[32487834];

// This function isn't currently used, but shows how you lookup
// a 7-card poker hand. pCards should be a pointer to an array
// of 7 integers each with value between 1 and 52 inclusive.
int LookupHand(int* pCards)
{
	int p = HR[53 + *pCards++];
	p = HR[p + *pCards++];
	p = HR[p + *pCards++];
	p = HR[p + *pCards++];
	p = HR[p + *pCards++];
	p = HR[p + *pCards++];
	return HR[p + *pCards++];
}

void LookupSingleHands()
{
	printf("Looking up individual hands...\n\n");

	// Create a 7-card poker hand (each card gets a value between 1 and 52)
	int cards[] = { 2, 6, 12, 14, 23, 26, 29 };
	int retVal = LookupHand(cards);
	printf("Category: %d\n", retVal >> 12);
	printf("Salt: %d\n", retVal & 0x00000FFF);
}


vector<uint32_t> partial_fisher_yates(uint32_t k, uint32_t n) {
	if (n < k) throw runtime_error("can't generate enough distinct elements in small interval");
		
	assert(n >= 1);
	vector<uint32_t> r;
	r.resize(n);
	for (uint32_t i = 0; i < n; ++i)
		r[i] = i;

	for (uint32_t i = 0; i < k; ++i) {		
		uint32_t z = i + rand() % (n-i);
		auto tmp = r[i];
		r[i] = r[z];
		r[z] = tmp;		
	}
	
	return r;
}



char StdDeck_rankChars[] = "23456789TJQKA";
char StdDeck_suitChars[] = "hdcs";

int tototo[8];

#define StdDeck_Rank_COUNT  13
#define StdDeck_Rank_FIRST  0
#define StdDeck_Rank_LAST   12
#define StdDeck_Suit_HEARTS   0
#define StdDeck_Suit_DIAMONDS 1
#define StdDeck_Suit_CLUBS    2
#define StdDeck_Suit_SPADES   3
#define StdDeck_Suit_FIRST    StdDeck_Suit_HEARTS
#define StdDeck_Suit_LAST     StdDeck_Suit_SPADES
#define StdDeck_RANK(index)  ((index) % StdDeck_Rank_COUNT)
#define StdDeck_SUIT(index)  ((index) / StdDeck_Rank_COUNT)
#define StdDeck_MAKE_CARD(rank, suit) ((suit * StdDeck_Rank_COUNT) + rank)

string cardToString(int cardIndex) {
	char r = StdDeck_rankChars[StdDeck_RANK(cardIndex)];
	char s_ch = StdDeck_suitChars[StdDeck_SUIT(cardIndex)];
	
	string s;
	s.push_back(r);
	s.push_back(s_ch);
	return s;
}


int stringToCardIndex(char *inString) {
	char *p;
	int rank, suit;

	p = inString;
	for (rank = StdDeck_Rank_FIRST; rank <= StdDeck_Rank_LAST; rank++)
		if (StdDeck_rankChars[rank] == toupper(*p))
			break;
	if (rank > StdDeck_Rank_LAST)
		return -1;
	++p;
	for (suit = StdDeck_Suit_FIRST; suit <= StdDeck_Suit_LAST; suit++)
		if (StdDeck_suitChars[suit] == tolower(*p))
			break;
	if (suit > StdDeck_Suit_LAST)
		return -1;
	//Return 1 to 52
	return suit * 13 + rank;

}


int _tmain(int argc, _TCHAR* argv[])
{
	printf("Testing the Two Plus Two 7-Card Evaluator\n");
	printf("-----------------------------------------\n\n");

	// Load the HandRanks.DAT file and map it into the HR array

	if (true) {
		printf("Loading HandRanks.DAT file...");
		memset(HR, 0, sizeof(HR));
		FILE * fin;
		fopen_s(&fin, "E:\\git\\poker\\XPokerEval\\XPokerEval.TwoPlusTwo\\HandRanks.dat", "rb");
		if (!fin)
			return false;
		size_t bytesread = fread(HR, sizeof(HR), 1, fin);	// get the HandRank Array
		fclose(fin);
		printf("complete.\n\n");
	}


	printf("%d\n", stringToCardIndex("2h"));
	printf("%d\n", stringToCardIndex("as"));

	int holeCard1 = stringToCardIndex("Jh");
	int holeCard2 = stringToCardIndex("7s");

	BoolArray taken(52, 0);

	taken.set(holeCard1);
	taken.set(holeCard2);

	vector<uint32_t> remaining_cards;
	for (int i = 0; i < 52; ++i)
	{
		if (taken.get(i) == false) {
			//store 1 - 52
			remaining_cards.push_back(i+1);
		}
	}

	int nPlayers = 5; //including hero

	int cardsNeeded = 4 * 2 + 5;

	int nSimulationsToDo = 1000000;
	
	int wonCount = 0;
	double tiedCount = 0;

	int distribution[13][52];

	for (int p = 0; p < 5; ++ p)
	{
		for (int c = 0; c < 52; ++c)
		{
			distribution[p][c] = 0;
		}
	}

	for (int i = 0; i < nSimulationsToDo; ++i)
	{

		//LookupSingleHands();
		vector<uint32_t> v = partial_fisher_yates(cardsNeeded, remaining_cards.size());

		for (int i = 0; i < 5; ++i) {
			//cout << "Picked #" << i << " as " << v[i] << " or " << remaining_cards[v[i]] << endl;
			//++distribution[i][v[i]];
		}

		int player_cards[5][7];

		player_cards[0][0] = holeCard1 + 1;
		player_cards[0][1] = holeCard2 + 1;

		
		for (int p = 0; p < nPlayers; ++ p)
		{
			for (int c=0; c<5;++c)
			{
				//Set common cards for everybody
				player_cards[p][2 + c] = remaining_cards[v[c]];
			}			
		}
		int random_card_index = 5;
		for (int p = 1; p < nPlayers; ++p)
		{
			player_cards[p][0] = remaining_cards[ v[random_card_index++] ];
			player_cards[p][1] = remaining_cards[ v[random_card_index++] ];
		}

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

		int scores[5];
		for (int p = 0; p < nPlayers; ++p)
		{
			

			scores[p] = LookupHand(player_cards[p]);
			//cout << "Score for " << p << " is " << scores[p] << endl;

			
		}

		int best_enemy_score = *std::max_element(scores + 1, scores + nPlayers);

		if (best_enemy_score < scores[0]) {
			++wonCount;
		}
		if (best_enemy_score == scores[0]) {
			//the tied equity depends on how many players tied
			int denom = 1;
			for (int j = 1; j < nPlayers; ++j) {
				if (scores[0] == scores[j]) {
					++denom;
				}
			}
			tiedCount += 1.0 / denom;
		}
		

		if (i < 0) 
		{
			for (int p = 0; p < nPlayers; ++p)
			{
				for (int c = 0; c < 7; ++c)
				{
					cout << "Player: " << p << " Card #: " << c << " = " <<
						player_cards[p][c] << " = " << 
						cardToString(player_cards[p][c]-1) << endl;
				}
			}
			/*for (auto const& chosen_card_index : v) {
				auto card_index = remaining_cards[chosen_card_index];
				cout << card_index << " is " << (card_index) << endl;
			}*/
		}
		
	}

	cout << "Did " << nSimulationsToDo << " Simulations" << endl;
	cout << "Won: " << wonCount << " " << setprecision(4) << (100.0 * wonCount / nSimulationsToDo) << "%" << endl;
	cout << "Tied: " << tiedCount << " " << setprecision(4) << (100.0 * tiedCount / nSimulationsToDo) << "%" << endl;
	cout << "Equity: " << setprecision(3) << (100.0 * (tiedCount+wonCount) / nSimulationsToDo) << "%" << endl;

	if (false)
	{
		for (int p = 0; p < nPlayers; ++p)
		{
			for (int c = 0; c < 52; ++c)
			{
				cout << "Player " << p << " number of hits for card " << c << " is " << distribution[p][c] << endl;
			}
		}
	}
	cout << "Press any key" << endl;
	std::cin.get();
	return 0;
}



