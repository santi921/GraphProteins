
        for chain in self.chains[:-1]:
            for res in chain.residues:
                remove_atoms = []
                for a in res.atoms:
                    if a.element.lower() == "h":
                        remove_atoms.append(a)
                        for b in a.bonds:
                            b.bonds.remove(a)

                for a in remove_atoms:
                    res.atoms.remove(a)

        if not self.sub_chain.residues:
            for res in self.chains[-1].residues:
                remove_atoms = []
                for a in res.atoms:
                    if a.element.lower() == "h":
                        remove_atoms.append(a)
                        for b in a.bonds:
                            b.bonds.remove(a)

                for a in remove_atoms:
                    res.atoms.remove(a)