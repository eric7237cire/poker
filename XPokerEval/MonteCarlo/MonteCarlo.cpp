// MonteCarlo.cpp : Defines the entry point for the console application.
//

//#include <windows.h>
#include <tchar.h>
#include "bool_array.h"
#include "mersenne.h"
#include <vector>
#include <string>
#include <iostream>
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

ZRandom zrand(42);

vector<uint32_t> generateUniformBitmap(uint32_t N, uint32_t Max) {
	if (Max < N) throw runtime_error("can't generate enough distinct elements in small interval");
	assert(Max >= 1);
	BoolArray bs(Max);
	uint32_t card = 0;
	while (card < N) {
		uint32_t v = zrand.getValue(Max - 1);
		if (!bs.get(v)) {
			bs.set(v);
			++card;
		}
	}
	vector<uint32_t> ans(N);
	bs.toArray(ans);
	return  ans;
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
	
	if (false) {
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
			remaining_cards.push_back(i);
		}
	}

	int nPlayers = 4;

	int cardsNeeded = 4 * 2 + 5;

	cardsNeeded = 50;

	//LookupSingleHands();
	vector<uint32_t> v = generateUniformBitmap(cardsNeeded, remaining_cards.size());

	for (auto const& chosen_card_index : v) {
		auto card_index = remaining_cards[chosen_card_index];
		cout << card_index << " is " << cardToString(card_index) << endl;
		
	}

	std::cin.get();
	return 0;
}



