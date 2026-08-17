# RIS System Bis

Implémentation simplifiée d'un estimateur de canal basé sur un MLP pour un RIS passif.

Le projet est organisé en trois dossiers dans `src`:

- `system_model`: géométrie, génération de canal et reconstruction;
- `data_generation`: création des jeux de données synthétiques;
- `ml_model`: réseau, entraînement, métriques et exécution.

La logique liée aux éléments actifs du RIS a été retirée.

## Exécution

```bash
python -m ris_system_bis.cli --train-samples 80000 --test-samples 20000
```

Ou via le point d'entrée installé:

```bash
ris-system-bis --train-samples 80000 --test-samples 20000
```
