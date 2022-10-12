def calculate_charge(sequence, charge_dict=aa_charge):
    """Calculates the charge of the peptide sequence at pH 7.4
    """
    sc_charges = [charge_dict.get(aa, 0) for aa in sequence]
    return sum(sc_charges)
