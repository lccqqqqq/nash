from torch._tensor import Tensor
from typing import List
import torch as t
import einops

class OpenMPS:
    def __init__(self, Psi: List[Tensor], orth_center: Tensor | None, forms: List[str] | None):

        # some logistics
        assert len(Psi) > 0, "MPS must have at least one tensor"
        self.d_phys = Psi[0].shape[0]
        self.n_sites = len(Psi)
        self.dtype = Psi[0].dtype
        self.device = Psi[0].device

        self.Psi = Psi
        self.orth_center = orth_center
        self.chis = self._get_chis()
        self.forms = forms

        self._assert_legal()
        self._assert_forms()

    def _get_chis(self) -> List[int]:
        return [psi.shape[1] for psi in self.Psi[1:]]

    def _assert_forms(self):
        inconsistent_form = False
        for i, (psi, form) in enumerate(zip(self.Psi, self.forms)):
            if form == 'left' or 'A':
                T = einops.einsum(
                    psi, psi.conj(), 
                    'd_phys chi_l chi_r, d_phys chi_l chi_rc -> chi_r chi_rc'
                )
                if not t.allclose(
                    T, t.eye(psi.shape[2], dtype=self.dtype, device=self.device)
                ):
                    inconsistent_form = True
                    break
            elif form == 'right' or 'B':
                T = einops.einsum(
                    psi, psi.conj(), 
                    'd_phys chi_l chi_r, d_phys chi_lc chi_r -> chi_l chi_lc'
                )
                if not t.allclose(
                    T, t.eye(psi.shape[1], dtype=self.dtype, device=self.device)
                ):
                    inconsistent_form = True
                    break
        
        if inconsistent_form:
            print(f"Warning: inconsistent forms detected: tensor {i} is not in form {form}, setting all forms to None")
            self.forms = None
        else:
            print(f"successfully created OpenMPS with {self.n_sites} sites with forms {self.forms}")


    def _assert_legal(self):
        """Validate that all tensors have consistent dtype, device, and physical dimension."""
        for i, tensor in enumerate(self.Psi):
            assert tensor.dtype == self.dtype, (
                f"Tensor at site {i} has dtype {tensor.dtype}, expected {self.dtype}"
            )
            assert tensor.device == self.device, (
                f"Tensor at site {i} is on device {tensor.device}, expected {self.device}"
            )
            assert tensor.shape[0] == self.d_phys, (
                f"Tensor at site {i} has physical dimension {tensor.shape[0]}, expected {self.d_phys}"
            )
        
        chis_right = [psi.shape[2] for psi in self.Psi[:-1]]
        assert all(chi == chis_right[i] for i, chi in enumerate(self.chis)), (
            f"Right bond dimensions are not consistent, found {chis_right} but expected {self.chis}"
        )
        assert self.chis[-1] == 1, f"Last bond dimension must be trivial, found {self.chis[-1]}"
        assert self.chis[0] == 1, f"First bond dimension must be trivial, found {self.chis[0]}"

        if self.forms is not None:
            assert self.orth_center is not None, f"Orthogonality center must be provided for mixed canonical form"
            assert t.norm(self.orth_center) == 1, f"Orthogonality center must be a unit vector containing (square roots of) Schmidt coefficients, found {self.orth_center}"

    def _reverse(self):
        self.Psi = [psi.transpose(-2, -1) for psi in self.Psi[::-1]]
        if self.forms is not None:
            self.forms = [_form.replace('A', 'B') if _form == 'B' else _form.replace('B', 'A') if _form == 'A' else _form for _form in self.forms[::-1]]

    def to_isometric_form(self, form: str = 'A'):
        if self.forms is not None and all(_form == form for _form in self.forms):
            return

        if form == 'B':
            self.Psi = [psi.transpose(-2, -1) for psi in self.Psi[::-1]]
            if self.forms is not None:
                self._reverse()

        psi = self.Psi[0] # shape: (d_phys, 1, chi)
        for j in range(len(self.Psi)):
            psi_grouped = einops.rearrange(psi, 'd_phys chi_l chi_r -> (d_phys chi_l) chi_r')
            left_iso, orth_center = t.linalg.qr(psi_grouped)
            self.Psi[j] = einops.rearrange(left_iso, '(d_phys chi_l) chi_r -> d_phys chi_l chi_r', d_phys=self.d_phys)
            if j < len(self.Psi) - 1:
                psi = einops.einsum(orth_center, self.Psi[j+1], 'chi_l bond, d_phys bond chi_r -> d_phys chi_l chi_r')
            else:
                # we are at the last site
                right_orth = orth_center.squeeze(-1)

        if form == 'right' or 'B':
            self.Psi = [psi.transpose(-2, -1) for psi in self.Psi[::-1]]

        self.chis = self._get_chis()
        self.orth_center = self.n_sites if form == 'left' or 'A' else 0
    

