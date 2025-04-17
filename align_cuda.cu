/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA+MPI assignment versions */
#include<cuda.h>
#include<mpi.h>


/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION( call )	{ cudaError_t check = call; if ( check != cudaSuccess ) fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }
#define CUDA_CHECK_KERNEL( )	{ cudaError_t check = cudaGetLastError(); if ( check != cudaSuccess ) fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }

/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1
#define NOT_FOUND_ULL 18446744073709551615ULL

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX	65535

/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"



/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
/* ADD KERNELS AND OTHER FUNCTIONS HERE */


/**	Kernel CUDA per trovare la prima occorrenza di ogni pattern in una sequenza
 * 
 * Input:	
 * 	sequence: 		Sequenza di caratteri da analizzare
 * 	pattern_buffer: Buffer continuo contenente tutti i pattern
 *  pat_offsets: 	Offset di ciascun pattern nel buffer
 * 	pat_length: 	Lunghezza di ciascun pattern
 * 	pat_found: 		Array per memorizzare la posizione minima trovata per ogni pattern
 * 	pat_number: 	Numero totale di pattern
 *  seq_length: 	Lunghezza della sequenza
 * 	rank: 			Rank del processo MPI
 * 	my_seq_length:	Lunghezza della porzione di sequenza per processo
 * */ 
__global__ void findPatterns(char *sequence, char *pattern_buffer, int *pat_offsets, unsigned long *pat_length, unsigned long long *pat_found, 
                             int pat_number, unsigned long seq_length, int rank, unsigned long my_seq_length) 
{
    extern __shared__ unsigned long long shared_min[];	// Shared memory per memorizzare il minimo locale del blocco

    const int pattern_idx = blockIdx.x;	// Indice del pattern (un pattern per blocco)
    const int local_start = threadIdx.x + blockIdx.y * blockDim.x;	// Posizione iniziale locale nel blocco
    const int global_start = local_start + (rank * my_seq_length);	// Posizione globale nella sequenza (aggiustata per il rank MPI)

    // --- Inizializzazione della shared memory ---
    if (threadIdx.x == 0) {
        shared_min[0] = (unsigned long long)NOT_FOUND;	// Il thread 0 del blocco inizializza il minimo locale a NOT_FOUND
    }
    __syncthreads(); // Sincronizza i thread *del blocco*

    // --- Controllo di validità ---
    if (pattern_idx >= pat_number || global_start + pat_length[pattern_idx] > seq_length) {
        return;	// Esce se il pattern non esiste o se la posizione supera la sequenza
    }

    // --- Confronto del pattern ---
    const int pattern_start = pat_offsets[pattern_idx];	// Offset del pattern corrente nel buffer continuo
    bool is_match = true; // Flag per indicare un match valido
    for (int i = 0; i < pat_length[pattern_idx]; i++) {
        if (sequence[global_start + i] != pattern_buffer[pattern_start + i]) {  // Confronta carattere per carattere
            is_match = false;
            break;
        }
    }

    // --- Aggiornamento del minimo locale ---
    // Se c'è un match, aggiorna il minimo nella shared memory
    if (is_match && (shared_min[0] == (unsigned long long)NOT_FOUND || global_start < shared_min[0])) {
        // Compare and Swap, serve per passare da uno stato indefinito (NOT_FOUND) a un valore valido senza conflitti. 
		// Permette di sapere se sei il primo thread a scrivere (tramite il valore restituito).
        unsigned long long old_min = atomicCAS(&shared_min[0], (unsigned long long)NOT_FOUND, (unsigned long long)global_start);
        // Serve per competere tra più thread che hanno valori validi, mantenendo solo il più piccolo
        if (old_min != (unsigned long long)NOT_FOUND && global_start < old_min) {
            atomicMin(&shared_min[0], (unsigned long long)global_start);
        }
    }
    __syncthreads(); // Sincronizza prima dell'aggiornamento globale

    // --- Scrittura del risultato globale ---
    // Solo il thread 0 del blocco aggiorna pat_found con il minimo locale
    if (threadIdx.x == 0 && shared_min[0] != (unsigned long long)NOT_FOUND) {
        unsigned long long old_found = atomicCAS(&pat_found[pattern_idx], (unsigned long long)NOT_FOUND, shared_min[0]); // Usa atomicCAS per inizializzare pat_found
        if (old_found != (unsigned long long)NOT_FOUND) {
            atomicMin(&pat_found[pattern_idx], shared_min[0]);	// Se già inizializzato, aggiorna con il valore minimo
        }
    }
}





/**	Kernel CUDA per aggiornare i contatori di match nella sequenza
 *  
 * Input:
 * 	pat_found: Posizioni minime dei pattern trovati (o NOT_FOUND_ULL se non trovati)
 * 	pat_length: Lunghezza di ciascun pattern
 * 	seq_matches: Array dei contatori di match per ogni posizione della sequenza
 * 	pat_number: Numero di pattern locali gestiti dal processo
 * 	seq_length: Lunghezza totale della sequenza
 * */ 
__global__ void updateSeqMatches(unsigned long long *pat_found, unsigned long *pat_length, int *seq_matches, int pat_number, unsigned long seq_length) 
{
    // Calcola l'indice del pattern per questo thread
    const int pattern_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Controllo di validità: esce se il pattern non esiste
    if (pattern_idx >= pat_number) {
        return;
    }

    // Ottiene la posizione del match per il pattern
    const unsigned long long match_pos = pat_found[pattern_idx];
    // Esce se il pattern non è stato trovato
    if (match_pos == NOT_FOUND_ULL) {
        return;
    }

    // Aggiorna i contatori di seq_matches per ogni posizione coperta dal pattern
    for (int i = 0; i < pat_length[pattern_idx]; i++) {
        // Incrementa atomicamente il contatore nella posizione match_pos + i
        atomicAdd(&seq_matches[match_pos + i], 1);
    }
}










/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate( rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length ) {

	/* Random length */
	unsigned long length = (unsigned long)rng_next_normal( random, (double)pat_rng_length_mean, (double)pat_rng_length_dev );
	if ( length > seq_length ) length = seq_length;
	if ( length <= 0 ) length = 1;

	/* Allocate pattern */
	char *pattern = (char *)malloc( sizeof(char) * length );
	if ( pattern == NULL ) {
		fprintf(stderr,"\n-- Error allocating a pattern of size: %lu\n", length );
		exit( EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence( rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
	unsigned long ind; 
	for( ind=0; ind<length; ind++ ) {
		double prob = rng_next( random );
		if( prob < prob_G ) seq[ind] = 'G';
		else if( prob < prob_C ) seq[ind] = 'C';
		else if( prob < prob_A ) seq[ind] = 'A';
		else seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Copy sample */
	unsigned long ind; 
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence( rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length ) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Regenerate sample */
	rng_t local_random = random_seq;
	rng_skip( &local_random, location );
	generate_rng_sequence( &local_random, prob_G, prob_C, prob_A, pattern, length);
}


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
	fprintf(stderr,"\n");
}



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	/* 0. Disattiva il buffering per stdout e stderr garantendo che i messaggi di debug o errore vengano scritti immediatamente	*/
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	/* 1. Read scenary arguments */
	
	/* 1.0. Init MPI before processing arguments */
	MPI_Init(&argc, &argv);	// Passare &argc, &argv è una buona pratica perché permette a MPI di gestire eventuali opzioni della riga di comando
	int rank, size;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	prob_C += prob_G;
	prob_A += prob_C;

	int pat_rng_num = atoi( argv[5] );
	unsigned long pat_rng_length_mean = atol( argv[6] );
	unsigned long pat_rng_length_dev = atol( argv[7] );
	
	int pat_samp_num = atoi( argv[8] );
	unsigned long pat_samp_length_mean = atol( argv[9] );
	unsigned long pat_samp_length_dev = atol( argv[10] );
	unsigned long pat_samp_loc_mean = atol( argv[11] );
	unsigned long pat_samp_loc_dev = atol( argv[12] );

	char pat_samp_mix = argv[13][0];
	if ( pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M' ) {
		fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	unsigned long seed = atol( argv[14] );

#ifdef DEBUG
	/* DEBUG: Print arguments */
	printf("\nArguments: seq_length=%lu\n", seq_length );
	printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
	printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
	printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
	printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
	printf("\n");
#endif // DEBUG


	int local_rank;		// Rank locale all'interno del nodo
	MPI_Comm local_comm;	// Comunicatore per i processi che condividono lo stesso nodo

	// Suddivisione di MPI_COMM_WORLD in sotto-comunicatori, uno per ogni nodo fisico
	// MPI_COMM_TYPE_SHARED raggruppa i processi MPI che condividono la stessa memoria (quindi lo stesso nodo)
	// Il valore `rank` è usato come "chiave" per mantenere l'ordine originale dei rank
	// MPI_INFO_NULL indica che non vengono passate informazioni aggiuntive
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
	MPI_Comm_rank(local_comm, &local_rank);	// Ottenimento del rank del processo all'interno del nodo locale

	int deviceCount;
	CUDA_CHECK_FUNCTION(cudaGetDeviceCount(&deviceCount));	// Numero di GPU disponibili sul nodo
	int gpu_id = local_rank % deviceCount; 	// Ogni processo MPI all'interno del nodo ottiene una GPU in modo ciclico usando il modulo (%)
	CUDA_CHECK_FUNCTION(cudaSetDevice(gpu_id));	// Impostazione della GPU da usare per questo processo MPI

	// DEBUG
	int dev_id;
	CUDA_CHECK_FUNCTION(cudaGetDevice(&dev_id));
	printf("MPI Rank %d (Local Rank %d) usa GPU %d\n", rank, local_rank, dev_id);


	/* 2. Initialize data structures */
	/* 2.1. Skip allocate and fill sequence */
	rng_t random = rng_new( seed );
	rng_skip( &random, seq_length );

	/* 2.2. Allocate and fill patterns */
	/* 2.2.1 Allocate main structures */
	int pat_number = pat_rng_num + pat_samp_num;
	unsigned long *pat_length = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	char **pattern = (char **)malloc( sizeof(char*) * pat_number );
	if ( pattern == NULL || pat_length == NULL ) {
		fprintf(stderr,"\n-- Error allocating the basic patterns structures for size: %d\n", pat_number );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 2.2.2 Allocate and initialize ancillary structure for pattern types */
	int ind;
	unsigned long lind;
	#define PAT_TYPE_NONE	0
	#define PAT_TYPE_RNG	1
	#define PAT_TYPE_SAMP	2
	char *pat_type = (char *)malloc( sizeof(char) * pat_number );
	if ( pat_type == NULL ) {
		fprintf(stderr,"\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_NONE;

	/* 2.2.3 Fill up pattern types using the chosen mode */
	switch( pat_samp_mix ) {
	case 'A':
		for( ind=0; ind<pat_rng_num; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		break;
	case 'B':
		for( ind=0; ind<pat_samp_num; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		break;
	default:
		if ( pat_rng_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		}
		else if ( pat_samp_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		}
		else if ( pat_rng_num < pat_samp_num ) {
			int interval = pat_number / pat_rng_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_RNG;
				else pat_type[ind] = PAT_TYPE_SAMP;
		}
		else {
			int interval = pat_number / pat_samp_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_SAMP;
				else pat_type[ind] = PAT_TYPE_RNG;
		}
	}

	/* 2.2.4 Generate the patterns */
	for( ind=0; ind<pat_number; ind++ ) {
		if ( pat_type[ind] == PAT_TYPE_RNG ) {
			pattern[ind] = pattern_allocate( &random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind] );
			generate_rng_sequence( &random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind] );
		}
		else if ( pat_type[ind] == PAT_TYPE_SAMP ) {
			pattern[ind] = pattern_allocate( &random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind] );
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
			rng_t random_seq_orig = rng_new( seed );
			generate_sample_sequence( &random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#else
			copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
		}
		else {
			fprintf(stderr,"\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind );
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}
	}
	free( pat_type );

	/* Allocate and move the patterns to the GPU */
	unsigned long *d_pat_length;
	char **d_pattern;
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pat_length, sizeof(unsigned long) * pat_number ) );	// Array che memorizza la lunghezza di ciascun pattern
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pattern, sizeof(char *) * pat_number ) );	// Array che punta a ciascun pattern

	// Array di puntatori a char * allocato sulla memoria host che fungerà da buffer per memorizzare i pattern prima di trasferirli sulla GPU
	char **d_pattern_in_host = (char **)malloc( sizeof(char*) * pat_number );
	if ( d_pattern_in_host == NULL ) {
		fprintf(stderr,"\n-- Error allocating the patterns structures replicated in the host for size: %d\n", pat_number );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/*
	* Per ogni pattern il codice:
	*     Alloca memoria sulla GPU per ciascun pattern (memoria per ogni singolo pattern di lunghezza pat_length[ind]).
    *     Copia il pattern dalla memoria host alla memoria device utilizzando cudaMemcpy, passando i dati da pattern[ind] (memoria host) 
	* 		a d_pattern_in_host[ind] (memoria device)
	*/
	for (ind=0; ind<pat_number; ind++) {
		CUDA_CHECK_FUNCTION(cudaMalloc(&(d_pattern_in_host[ind]), sizeof(char *) * pat_length[ind]));
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern_in_host[ind], pattern[ind], pat_length[ind] * sizeof(char), cudaMemcpyHostToDevice));
	}

	/*
	* Una volta che tutti i pattern sono stati allocati e copiati nella memoria device, 
	* copia l'array di puntatori d_pattern_in_host (memoria host) nell'array di puntatori d_pattern (memoria device), 
	* permettendo al codice CUDA di accedere ai pattern sulla GPU
	*/
	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern, d_pattern_in_host, pat_number * sizeof(char *), cudaMemcpyHostToDevice ) );

	/* Avoid the usage of arguments to take strategic decisions
	 * In a real case the user only has the patterns and sequence data to analize
	 */
	argc = 0;
	argv = NULL;
	pat_rng_num = 0;
	pat_rng_length_mean = 0;
	pat_rng_length_dev = 0;
	pat_samp_num = 0;
	pat_samp_length_mean = 0;
	pat_samp_length_dev = 0;
	pat_samp_loc_mean = 0;
	pat_samp_loc_dev = 0;
	pat_samp_mix = '0';

	/* 2.3. Other result data and structures */
	int pat_matches = 0;

	/* 2.3.1. Other results related to patterns */
	unsigned long long *pat_found;
	pat_found = (unsigned long long*)malloc( sizeof(unsigned long long) * pat_number );
	if ( pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	
	/* 3. Start global timer */
	// Le barriere servono poichè potrebbe esserci un rischio che il timer venga avviato prima che tutte le operazioni siano effettivamente completate, distorcendo la misurazione del tempo */
    CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );	// Evita lavoro GPU pendente prima di sincronizzarsi con MPI
	MPI_Barrier( MPI_COMM_WORLD );	// Tutti i processi si allineano prima di partire
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
	/* 2.1. Allocate and fill sequence */
	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	random = rng_new( seed );
	generate_rng_sequence( &random, prob_G, prob_C, prob_A, sequence, seq_length);

#ifdef DEBUG
	/* DEBUG: Print sequence and patterns */
	printf("-----------------\n");
	printf("Sequence: ");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( "%c", sequence[lind] );
	printf("\n-----------------\n");
	printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num );
	int debug_pat;
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( "Pat[%d]: ", debug_pat );
		for( lind=0; lind<pat_length[debug_pat]; lind++ ) 
			printf( "%c", pattern[debug_pat][lind] );
		printf("\n");
	}
	printf("-----------------\n\n");
#endif // DEBUG

	/* 2.3.2. Other results related to the main sequence */
	int *seq_matches;
	seq_matches = (int *)malloc( sizeof(int) * seq_length );
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}


	/* 4. Initialize ancillary structures */
	for( ind=0; ind<pat_number; ind++) {
		pat_found[ind] = (unsigned long long)NOT_FOUND;
	}
	for( lind=0; lind<seq_length; lind++) {
		seq_matches[lind] = NOT_FOUND;
	}



	/* 5. Search for each pattern */


	// --- Preparazione del buffer concatenato per i pattern ---

	// Calcola la lunghezza totale di tutti i pattern
	int total_pattern_length = 0;
	for (int i = 0; i < pat_number; i++) {
		total_pattern_length += pat_length[i];
	}

	// Alloca memoria sull'host per il buffer concatenato e gli offset
	char *h_pattern_buffer = (char*)malloc(total_pattern_length * sizeof(char));
	int *h_pattern_offsets = (int*)malloc(pat_number * sizeof(int));
	if (!h_pattern_buffer || !h_pattern_offsets) {
		fprintf(stderr, "Errore: Allocazione memoria host fallita\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Riempie il buffer concatenato con i pattern e calcola gli offset
	int current_offset = 0;
	for (int i = 0; i < pat_number; i++) {
		memcpy(h_pattern_buffer + current_offset, pattern[i], pat_length[i] * sizeof(char));
		h_pattern_offsets[i] = current_offset;
		current_offset += pat_length[i];
	}

	// --- Allocazione memoria sulla GPU ---

	// Alloca buffer per i pattern e gli offset sul device
	char *d_pattern_buffer;
	int *d_pattern_offsets;
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pattern_buffer, total_pattern_length * sizeof(char)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pattern_offsets, pat_number * sizeof(int)));

	// --- Copia dei dati dalla CPU alla GPU ---

	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern_buffer, h_pattern_buffer, total_pattern_length * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern_offsets, h_pattern_offsets, pat_number * sizeof(int), cudaMemcpyHostToDevice));

	// --- Allocazione memoria per il kernel findPatterns ---

	// Allocazione array sul device
	char *d_sequence;               // Sequenza di caratteri
	int *d_seq_matches;             // Contatore di match per posizione
	unsigned long long *d_pat_found; // Posizioni minime dei pattern
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_sequence, seq_length * sizeof(char)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pat_found, pat_number * sizeof(unsigned long long)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_seq_matches, seq_length * sizeof(int)));

	// Allocazione array sull'host
	unsigned long long *h_pat_found = (unsigned long long*)malloc(pat_number * sizeof(unsigned long long));
	int *h_seq_matches = NULL;
	if (rank == 0) {
		h_seq_matches = (int*)malloc(seq_length * sizeof(int));
	}
	if (!h_pat_found || (rank == 0 && !h_seq_matches)) {
		fprintf(stderr, "Errore: Allocazione memoria host fallita\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// --- Inizializzazione memoria sul device ---

	CUDA_CHECK_FUNCTION(cudaMemcpy(d_sequence, sequence, seq_length * sizeof(char), cudaMemcpyHostToDevice));	// Copia la sequenza dall'host al device
	CUDA_CHECK_FUNCTION(cudaMemset(d_pat_found, NOT_FOUND, pat_number * sizeof(unsigned long long)));	// Inizializza d_pat_found con NOT_FOUND
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, pat_number * sizeof(unsigned long), cudaMemcpyHostToDevice)); // Copia le lunghezze dei pattern dall'host al device

	// Inizializzazione d_seq_matches: -1 per rank 0, 0 per altri rank
	// Nota: rank 0 usa -1 per evitare conteggi doppi dei match nelle porzioni della sequenza
	if (rank == 0) {
		CUDA_CHECK_FUNCTION(cudaMemset(d_seq_matches, -1, seq_length * sizeof(int)));
	} else {
		CUDA_CHECK_FUNCTION(cudaMemset(d_seq_matches, 0, seq_length * sizeof(int)));
	}

	// --- Configurazione e lancio del kernel findPatterns ---

	// Parametri del kernel
	const dim3 blockSize(128); // Thread per blocco 
	const unsigned long my_seq_length = seq_length / size; // Porzione di sequenza per processo
	const size_t maxGridY = 65535; // Limite massimo CUDA per dimensione griglia Y
	size_t gridY = (my_seq_length + blockSize.x - 1) / blockSize.x; // Blocchi necessari per coprire my_seq_length
	gridY = (gridY > maxGridY) ? maxGridY : gridY; // Rispetta il limite CUDA
	const size_t sharedMemSize = sizeof(unsigned long long); // 8 byte per il minimo locale

	// Lancia il kernel findPatterns
	findPatterns<<<dim3(pat_number, gridY), blockSize, sharedMemSize>>>(
		d_sequence, d_pattern_buffer, d_pattern_offsets, d_pat_length, d_pat_found, pat_number, seq_length, rank, my_seq_length);

	// --- Post-elaborazione dei risultati ---

	// Copia i risultati di d_pat_found sull'host (l'operazione non comincia fino al completamento del kernel poiché riconosce il trasferimento di dati)
	CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, pat_number * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

	// Riduce i minimi locali di tutti i processi per ottenere il minimo globale
	MPI_Allreduce(pat_found, h_pat_found, pat_number, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

	// Conta il numero di pattern trovati (solo su CPU per semplicità)
	for (int j = 0; j < pat_number; j++) {
		if (h_pat_found[j] != NOT_FOUND_ULL) {
			pat_matches++;
		}
	}




	// --- Configurazione del secondo kernel: updateSeqMatches --- 

	const int local_pat_number = (rank < pat_number % size) ? (pat_number / size + 1) : (pat_number / size); // Calcola il numero di pattern assegnati a questo processo MPI 
	const int start_pat = (rank < pat_number % size) ? (rank * local_pat_number) : (rank * (pat_number / size) + pat_number % size); // Calcola l'indice del primo pattern assegnato a questo processo

	// Alloca un array sul device per la porzione locale di h_pat_found (necessario perché d_pat_found contiene solo i dati locali del processo)
	unsigned long long *d_pat_found_split;
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pat_found_split, local_pat_number * sizeof(unsigned long long)));
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_found_split, &h_pat_found[start_pat], local_pat_number * sizeof(unsigned long long), cudaMemcpyHostToDevice));

	// Configura i parametri del kernel 
	const dim3 blockSize2(256); // Numero di thread per blocco
	const dim3 gridSize2((local_pat_number + blockSize2.x - 1) / blockSize2.x); // Numero di blocchi per coprire tutti i pattern locali. Griglia 1D. 

	// Lancia il kernel updateSeqMatches per aggiornare d_seq_matches 
	updateSeqMatches<<<gridSize2, blockSize2>>>(d_pat_found_split, &d_pat_length[start_pat], d_seq_matches, local_pat_number, seq_length);

	// --- Post-elaborazione dei risultati --- 

	// Copia d_seq_matches dall'host alla CPU. Nota: cudaMemcpy blocca, quindi non serve cudaDeviceSynchronize 
	CUDA_CHECK_FUNCTION(cudaMemcpy(seq_matches, d_seq_matches, seq_length * sizeof(int), cudaMemcpyDeviceToHost));

	// Riduce i contatori locali di tutti i processi in h_seq_matches (solo rank 0) 
	MPI_Reduce(seq_matches, h_seq_matches, seq_length, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// --- Deallocazione memoria --- 

	// Libera gli array allocati sul device 
	CUDA_CHECK_FUNCTION(cudaFree(d_sequence));
	CUDA_CHECK_FUNCTION(cudaFree(d_pat_found));
	CUDA_CHECK_FUNCTION(cudaFree(d_seq_matches));
	CUDA_CHECK_FUNCTION(cudaFree(d_pat_found_split));
	CUDA_CHECK_FUNCTION(cudaFree(d_pat_length));
	CUDA_CHECK_FUNCTION(cudaFree(d_pattern_buffer));
	CUDA_CHECK_FUNCTION(cudaFree(d_pattern_offsets));

	// Libera l'array di puntatori ai pattern sul device 
	for (int ind = 0; ind < pat_number; ind++) {
		CUDA_CHECK_FUNCTION(cudaFree(d_pattern_in_host[ind]));
	}
	CUDA_CHECK_FUNCTION(cudaFree(d_pattern));

	// Libera i buffer allocati sull'host 
	free(d_pattern_in_host);
	free(h_pattern_buffer);
	free(h_pattern_offsets);


	/* 7. Calcolo dei checksum e deallocazione memoria finale */
	unsigned long checksum_matches = 0;
	unsigned long checksum_found = 0;

	// Calcola i checksum solo sul processo master (rank 0)
	if (rank == 0) {
		// Somma le posizioni valide in h_pat_found
		for (int i = 0; i < pat_number; i++) {
			if (h_pat_found[i] != (unsigned long)NOT_FOUND) {
				checksum_found += h_pat_found[i];
			}
		}
		checksum_found %= CHECKSUM_MAX; 

		// Somma i valori validi in h_seq_matches
		for (unsigned long i = 0; i < seq_length; i++) {
			if (h_seq_matches[i] != NOT_FOUND) {
				checksum_matches += h_seq_matches[i];
			}
		}
		checksum_matches %= CHECKSUM_MAX; 

		free(h_seq_matches);	// Libera la memoria di h_seq_matches
	}
	free(h_pat_found); // Libera la memoria di h_pat_found (tutti i processi)


#ifdef DEBUG
	/* DEBUG: Write results */
	printf("-----------------\n");
	printf("Found start:");
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( " %lu", pat_found[debug_pat] );
	}
	printf("\n");
	printf("-----------------\n");
	printf("Matches:");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( " %d", seq_matches[lind] );
	printf("\n");
	printf("-----------------\n");
#endif // DEBUG

	/* Free local resources */	
	free( sequence );
	free( seq_matches );

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 8. Stop global timer */
    CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	MPI_Barrier( MPI_COMM_WORLD );
	ttotal = cp_Wtime() - ttotal;

	if (rank == 0) {
		/* 9. Output for leaderboard */
		printf("\n");
		/* 9.1. Total computation time */
		printf("Time: %lf\n", ttotal );

		/* 9.2. Results: Statistics */
		printf("Result: %d, %lu, %lu\n\n", 
				pat_matches,
				checksum_found,
				checksum_matches );
	}
		
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );


	/* 11. End */
	MPI_Finalize();
	return 0;
}	
