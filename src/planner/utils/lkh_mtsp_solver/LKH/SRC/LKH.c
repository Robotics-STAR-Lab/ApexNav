#include "LKH.h"

/* All global variables of the problem. */

int AscentCandidates;   /* Number of candidate edges to be associated
                           with each node during the ascent */
int Asymmetric;         /* Is the instance asymmetric? */
int BackboneTrials;     /* Number of backbone trials in each run */
int Backtracking;       /* Specifies whether backtracking is used for 
                           the first move in a sequence of moves */
GainType BestCost;      /* Cost of the tour in BestTour */
GainType BestPenalty;   /* Penalty of the tour in BestTour */
int *BestTour;          /* Table containing best tour found */
GainType BetterCost;    /* Cost of the tour stored in BetterTour */
GainType BetterPenalty; /* Penalty of the tour stored in BetterTour */
int *BetterTour;        /* Table containing the currently best tour 
                           in a run */
int BWTSP_B;            /* Number of black nodes in a BWTSP instance */ 
int BWTSP_Q;            /* Maximum number of subsequent white nodes in a
                           BWTSP instance */ 
int BWTSP_L;            /* Maximum length of any path between two black nodes
                           in a BTWSP instance */
int CacheMask;  /* Mask for indexing the cache */
int *CacheVal;  /* Table of cached distances */
int *CacheSig;  /* Table of the signatures of cached 
                   distances */
int CandidateFiles;     /* Number of CANDIDATE_FILEs */
int EdgeFiles;          /* Number of EDGE_FILEs */
int *CostMatrix;        /* Cost matrix */
GainType CurrentGain;
GainType CurrentPenalty;
int DemandDimension;    /* Number of commodities in a M-PDTSP instance */
Node *Depot;
int Dimension;          /* Number of nodes in the problem */
int DimensionSaved;     /* Saved value of Dimension */
int Dim;                /* DimensionSaved - Salesmen + 1 */
double DistanceLimit;   /* Maxixim route distance for a CVRP instance */
double Excess;          /* Maximum alpha-value allowed for any 
                           candidate edge is set to Excess times the 
                           absolute value of the lower bound of a 
                           solution tour */
int ExtraCandidates;    /* Number of extra neighbors to be added to 
                           the candidate set of each node */
Node *FirstActive, *LastActive; /* First and last node in the list 
                                   of "active" nodes */
Constraint *FirstConstraint; /* First constraint in the list of SOP 
                                precedence constraints */
Node *FirstNode;        /* First node in the list of nodes */
Segment *FirstSegment;  /* A pointer to the first segment in the cyclic 
                           list of segments */
SSegment *FirstSSegment;        /* A pointer to the first super segment in
                                   the cyclic list of segments */
int Gain23Used; /* Specifies whether Gain23 is used */
int GainCriterionUsed;  /* Specifies whether L&K's gain criterion is 
                           used */
double GridSize;        /* Grid size used in toroidal instances */
int GroupSize;  /* Desired initial size of each segment */
int SGroupSize; /* Desired initial size of each super segment */
int Groups;     /* Current number of segments */
int SGroups;    /* Current number of super segments */
unsigned Hash;  /* Hash value corresponding to the current tour */
Node **Heap;    /* Heap used for computing minimum spanning trees */
HashTable *HTable;      /* Hash table used for storing tours */
int InitialPeriod;      /* Length of the first period in the ascent */
int InitialStepSize;    /* Initial step size used in the ascent */
double InitialTourFraction;     /* Fraction of the initial tour to be 
                                   constructed by INITIAL_TOUR_FILE edges */
int Kicks;      /* Specifies the number of K-swap-kicks */
int KickType;   /* Specifies K for a K-swap-kick */
char *LastLine; /* Last input line */
double LowerBound;      /* Lower bound found by the ascent */
int M;          /* The M-value is used when solving an ATSP-
                   instance by transforming it to a STSP-instance */
int MaxBreadth; /* The maximum number of candidate edges 
                   considered at each level of the search for
                   a move */
int MaxCandidates;      /* Maximum number of candidate edges to be 
                           associated with each node */
double DistanceLimit; /* Maxixim route distance for a CVRP instance */
int MaxMatrixDimension; /* Maximum dimension for an explicit cost matrix */
int MaxSwaps;   /* Maximum number of swaps made during the 
                   search for a move */
int MaxTrials;  /* Maximum number of trials in each run */
int MergeTourFiles;     /* Number of MERGE_TOUR_FILEs */
int MoveType;   /* Specifies the sequantial move type to be used 
                   in local search. A value K >= 2 signifies 
                   that a k-opt moves are tried for k <= K */
int MoveTypeSpecial; /* A special (3- or 5-opt) move is used */
Node *NodeSet;  /* Array of all nodes */
int Norm;       /* Measure of a 1-tree's discrepancy from a tour */
int NonsequentialMoveType;      /* Specifies the nonsequential move type to
                                   be used in local search. A value 
                                   L >= 4 signifies that nonsequential
                                   l-opt moves are tried for l <= L */
GainType Optimum;       /* Known optimal tour length. 
                           If StopAtOptimum is 1, a run will be 
                           terminated as soon as a tour length 
                           becomes equal this value */
int PatchingA;  /* Specifies the maximum number of alternating
                   cycles to be used for patching disjunct cycles */
int PatchingC;  /* Specifies the maximum number of disjoint cycles to be 
                   patched (by one or more alternating cycles) */
GainType PenaltyGain;
int Precision;  /* Internal precision in the representation of 
                   transformed distances */
int PredSucCostAvailable; /* PredCost and SucCost are available */
int POPMUSIC_InitialTour;  /* Specifies whether the first POPMUSIC tour
                              is used as initial tour for LK */
int POPMUSIC_MaxNeighbors; /* Maximum number of nearest neighbors used 
                              as candidates in iterated 3-opt */
int POPMUSIC_SampleSize;   /* The sample size */
int POPMUSIC_Solutions;    /* Number of solutions to generate */
int POPMUSIC_Trials;       /* Maximum trials used for iterated 3-opt */
unsigned *Rand;           /* Table of random values */
int Recombination;        /* IPT or GPX2 */
int RestrictedSearch;     /* Specifies whether the choice of the first
                             edge to be broken is restricted */
short Reversed; /* Boolean used to indicate whether a tour has 
                   been reversed */
int Run;        /* Current run number */
int Runs;       /* Total number of runs */
int Scale;      /* Scale factor for Euclidean and ATT instances */ 
double ServiceTime; /* Service time for a CVRP instance */
int Serial;
unsigned Seed;  /* Initial seed for random number generation */
double StartTime;       /* Time when execution starts */
int StopAtOptimum;      /* Specifies whether a run will be terminated if 
                           the tour length becomes equal to Optimum */
int Subgradient;        /* Specifies whether the Pi-values should be 
                           determined by subgradient optimization */
int SubproblemSize;     /* Number of nodes in a subproblem */
int SubsequentMoveType; /* Specifies the move type to be used for all 
                           moves following the first move in a sequence 
                           of moves. The value K >= 2 signifies that a 
                           K-opt move is to be used */
int SubsequentMoveTypeSpecial; /* A special (3- or 5-opt) subsequent move
                                  is used */
int SubsequentPatching; /* Species whether patching is used for 
                           subsequent moves */
SwapRecord *SwapStack;  /* Stack of SwapRecords */
int Swaps;      /* Number of swaps made during a tentative move */
int OldSwaps;   /* Saved number of swaps */
double TimeLimit;       /* The time limit in seconds */
int TotalDemand;        /* Sum of demands for a CVRP instance */
int TraceLevel; /* Specifies the level of detail of the output 
                   given during the solution process. 
                   The value 0 signifies a minimum amount of 
                   output. The higher the value is the more 
                   information is given */
int Trial;      /* Ordinal number of the current trial */
GainType TSPTW_CurrentMakespanCost;
int TSPTW_Makespan;

/* The following variables are read by the functions ReadParameters and 
   ReadProblem: */

char *ParameterFileName, *ProblemFileName, *PiFileName,
    *TourFileName, *OutputTourFileName, *InputTourFileName,
    **CandidateFileName, **EdgeFileName, *InitialTourFileName,
    *SubproblemTourFileName, **MergeTourFileName,
    *MTSPSolutionFileName, *SINTEFSolutionFileName;
char *Name, *Type, *EdgeWeightType, *EdgeWeightFormat,
    *EdgeDataFormat, *NodeCoordType, *DisplayDataType;
int CandidateSetSymmetric, CandidateSetType, Capacity,
    CoordType, DelaunayPartitioning, DelaunayPure,
    ExternalSalesmen,
    ExtraCandidateSetSymmetric, ExtraCandidateSetType,
    InitialTourAlgorithm,
    KarpPartitioning, KCenterPartitioning, KMeansPartitioning,
    MTSPDepot, MTSPMinSize, MTSPMaxSize, MTSPObjective,
    MoorePartitioning,
    PatchingAExtended, PatchingARestricted,
    PatchingCExtended, PatchingCRestricted,
    ProblemType, RiskThreshold,
    RohePartitioning, Salesmen, SierpinskiPartitioning,
    SubproblemBorders, SubproblemsCompressed, WeightType, WeightFormat;

FILE *ParameterFile, *ProblemFile, *PiFile, *InputTourFile,
    *InitialTourFile, *SubproblemTourFile, **MergeTourFile;
CostFunction Distance, D, C, c, OldDistance;
MoveFunction BestMove, BacktrackMove, BestSubsequentMove;
PenaltyFunction Penalty;
