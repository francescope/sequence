/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * OpenMP version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.2
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>


/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1

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
 *
 */


/* Questa funzione aggiorna l'array seq_matches, che tiene traccia del numero di volte in cui un pattern è stato trovato in una sequenza.
*
* Parametri:
*    - int pat: 					l'indice del pattern trovato
*    - unsigned long *pat_found: 	array che memorizza la posizione iniziale di ciascun pattern nella sequenza (pat_found[pat] contiene la minore posizione in cui è stato trovato il pattern pat)
*    - unsigned long *pat_length: 	array che contiene la lunghezza di ciascun pattern (pat_length[pat] è la lunghezza del pattern pat)
*    - int *seq_matches: 			array che memorizza, per ogni posizione della sequenza, il numero di pattern che iniziano in quella posizione
*/
void increment_matches( int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches ) {
	unsigned long ind;	
	#pragma omp parallel for schedule(static)	// Non c'è bisogno di sincronizzazione perché ogni thread lavora su un indice diverso dell'array
	for( ind=0; ind<pat_length[pat]; ind++) {
		if ( seq_matches[ pat_found[pat] + ind ] == NOT_FOUND )
			seq_matches[ pat_found[pat] + ind ] = 0;
		else
			seq_matches[ pat_found[pat] + ind ] ++;
	}
}

/*
 * Questa funzione genera una sequenza casuale di caratteri ('G', 'C', 'A', 'T') 
 * utilizzando un generatore di numeri casuali.
 *
 * Parametri:
 * - rng_t *random: 					puntatore allo stato iniziale del generatore casuale
 * - float prob_G, prob_C, prob_A: 		probabilità cumulative per 'G', 'C' e 'A'
 * - char *seq: 						puntatore all'array in cui verrà salvata la sequenza generata
 * - unsigned long length: 				lunghezza della sequenza da generare
 */
void generate_rng_sequence(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {

    unsigned int n_threads = omp_get_max_threads();	// Ottengo il numero massimo di thread disponibili

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();	// Ottengo l'ID del thread corrente

		// Calcola la porzione della sequenza che questo thread deve elaborare
      
	    // Ogni thread lavora su un "chunk" distinto ed equo
        unsigned long chunk_size = (length + n_threads - 1) / n_threads; 
		
		// Ogni thread elabora una porzione (da start a end) senza sovrapposizione
        unsigned long start = tid * chunk_size;
        unsigned long end = (start + chunk_size < length) ? start + chunk_size : length;

        // Copia lo stato iniziale del generatore casuale
        rng_t thread_state = *random;

		// Salta lo stato del generatore casuale fino alla posizione di inizio del chunk
        // Ogni thread inizia dal punto corretto per evitare sovrapposizioni
        rng_skip(&thread_state, start);		

        // Generazione casuale nella porzione assegnata
        for (unsigned long ind = start; ind < end; ind++) {
			// Genera un numero casuale tra 0 e 1
            double prob = rng_next(&thread_state);

            // Assegna un carattere basato sulle probabilità cumulative
            if (prob < prob_G) seq[ind] = 'G';			// Probabilità per 'G'
            else if (prob < prob_C) seq[ind] = 'C';		// Probabilità per 'C'
            else if (prob < prob_A) seq[ind] = 'A';		// Probabilità per 'A'
            else seq[ind] = 'T';						// Il resto va a 'T'
        }
    }

    // Avanza lo stato globale del generatore casuale per riflettere l'intero processo
    // Questo consente di preservare la continuità dei numeri casuali se la funzione viene chiamata di nuovo
    rng_skip(random, length);
}


/* 
 * Questa funzione estrae un campione di lunghezza `length` da una sequenza `sequence` a una posizione 
 * casuale generata tramite una distribuzione normale, e lo copia nel pattern.
 * 
 * Parametri:
 * - rng_t *random: 					puntatore allo stato iniziale del generatore casuale
 * - char *sequence: 					la sequenza di origine da cui copiare i dati
 * - unsigned long seq_length: 			la lunghezza della sequenza di origine
 * - unsigned long pat_samp_loc_mean: 	la media della posizione di inizio del campione
 * - unsigned long pat_samp_loc_dev: 	la deviazione standard della posizione di inizio del campione
 * - char *pattern: 					la sequenza di destinazione in cui sarà copiato il campione
 * - unsigned long length: 				la lunghezza del campione da copiare
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	// "Estrai" la posizione di inizio del campione
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );

    // Se la posizione generata è troppo vicina alla fine della sequenza e non consente di copiare un campione della lunghezza richiesta,
    // allora la posizione viene limitata per evitare di uscire dai limiti della sequenza (stessa cosa per i limiti negativi).
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	// Ogni thread copia una parte del campione nella sequenza `pattern`
	unsigned long ind; 
	#pragma omp parallel for schedule(static)	// Non c'è bisogno di sincronizzazione perché ogni thread lavora su un indice diverso dell'array
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
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
	/* 0. Default output and error without buffering, forces to write immediately */
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	/* 1. Read scenary arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
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
		exit( EXIT_FAILURE );
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
		exit( EXIT_FAILURE );
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
		exit( EXIT_FAILURE );
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
			exit( EXIT_FAILURE );
		}
	}
	free( pat_type );

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
	unsigned long *pat_found;
	pat_found = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	if ( pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	
	/* 3. Start global timer */
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	/* 2.1. Allocate and fill sequence */
	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
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
		exit( EXIT_FAILURE );
	}

	/* 4. Initialize ancillary structures */
	#pragma omp parallel
	{
		#pragma omp for schedule(static)
		for( ind=0; ind<pat_number; ind++) {
			pat_found[ind] = (unsigned long)NOT_FOUND;
		}
		#pragma omp for schedule(static)
		for( lind=0; lind<seq_length; lind++) {
			seq_matches[lind] = NOT_FOUND;
		}
	}

	
	// 5. Search for each pattern 
	unsigned long start;				//init
	int pat;							//init
	// Ciclo più esterno parallelizzato, ogni thread lavora su un pattern separato in modo indipendente dagli altri
	#pragma omp parallel for schedule(static)
	for (pat = 0; pat < pat_number; pat++) {
		
		unsigned long local_found = NOT_FOUND;	// Dichiarandola all'interno del ciclo parallelo ogni thread avrà la propria copia

		// Ciclo intermedio (sequenziale) (itera su tutte le possibili posizioni di partenza della sequenza per il pattern corrente)
		for (start = 0; start <= seq_length - pat_length[pat]; start++) {
			// Ciclo interno (sequenziale) (itera sugli elementi del pattern e verifica la corrispondenza con la sequenza)
			for (lind = 0; lind < pat_length[pat]; lind++) {
				// Condizione di uscita dal ciclo se i nucleotidi non corrispondono
				if (sequence[start + lind] != pattern[pat][lind]) {
					break;
				}
			}

			// Se il pattern è stato trovato (quindi se l'indice del ciclo arriva alla lunghezza del pattern), esci dal ciclo
			if (lind == pat_length[pat]) {
				local_found = start;
				break;
			}
		}

		// Se il pattern è stato trovato, aggiorna l'indice e incrementa i match
		if (local_found != NOT_FOUND) {
			pat_found[pat] = local_found;
			#pragma omp critical
			{
				pat_matches++;	//ha bisogno di essere in una sezione critica poiché è una variabile che viene aggiornata da tutti i threads
				
				/* Incrementa i match nel caso il pattern sia stato trovato,
					sta in una sezione critica poiché diversi pattern potrebbero andare ad aggiornare le stesse entry di seq_matches */
				increment_matches(pat, pat_found, pat_length, seq_matches);
			}
		}
	}


	// 7. Check sums 

	// Checksum delle posizioni trovate (checksum_found)
	unsigned long checksum_found = 0;
	#pragma omp parallel
	{
		unsigned long local_sum = 0;  // Somma locale per ogni thread
		#pragma omp for nowait
		for (int ind = 0; ind < pat_number; ind++) {
			if (pat_found[ind] != (unsigned long)NOT_FOUND) {
				local_sum += pat_found[ind];  
			}
		}
		#pragma omp atomic
		checksum_found += local_sum;  // Combina i risultati dei thread
	}
	checksum_found %= CHECKSUM_MAX;  // Operazione modulo una sola volta alla fine


	// Checksum dei match nella sequenza (checksum_matches)
	unsigned long checksum_matches = 0;
	#pragma omp parallel
	{
		unsigned long local_sum = 0;  // Somma locale per ogni thread
		#pragma omp for nowait
		for( lind = 0; lind < seq_length; lind++) {
			if (seq_matches[lind] != NOT_FOUND)
				local_sum += seq_matches[lind];
			}
		#pragma omp atomic
		checksum_matches += local_sum;  // Combina i risultati dei thread
	}
	checksum_matches %= CHECKSUM_MAX;  // Operazione modulo una sola volta alla fine



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
	ttotal = cp_Wtime() - ttotal;

	/* 9. Output for leaderboard */
	printf("\n");
	/* 9.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 9.2. Results: Statistics */
	printf("Result: %d, %lu, %lu\n\n", 
			pat_matches,
			checksum_found,
			checksum_matches );
		
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );

	/* 11. End */
	return 0;
}