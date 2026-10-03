#include "../dgemm_cli.h"

int main(int argc, char **argv)
{
    return dgemm_cli_run(argc, argv, "avx2");
}
