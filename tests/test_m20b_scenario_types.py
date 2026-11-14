"""Scenario revisions must not erase custom behavior outside the qualified model."""
import pytest

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec as Spec,
    UniformElectricField,
)
from conductor_sim.scenarios import PrescribedScenario


class CustomField(PrescribedField):
    def electric_field_at(self, point_m):
        return tuple(2 * x for x in super().electric_field_at(point_m))

    def potential_at(self, point_m):
        return 2 * super().potential_at(point_m)


class CustomParticle(PointCharge):
    pass


class CustomSpec(Spec):
    pass


@pytest.mark.parametrize('custom', ['field', 'particle', 'spec'])
def test_scenario_rejects_unqualified_model_subclasses(custom):
    field_type = CustomField if custom == 'field' else PrescribedField
    particle_type = CustomParticle if custom == 'particle' else PointCharge
    spec_type = CustomSpec if custom == 'spec' else Spec
    spec = spec_type(
        field_type((UniformElectricField((2., 0., 0.)),)),
        particle_type(2.), 4., (0., 0., 0.), .2, .02,
    )
    # The legacy spec permits these subclasses, but M20b reconstruction cannot
    # preserve arbitrary subclass physics or state in its bounded audit schema.
    with pytest.raises(TypeError):
        PrescribedScenario('custom', spec, (('external', 'External field'),),
                           'Unsupported custom model behavior.')
