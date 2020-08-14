#include <mitsuba/core/fwd.h>
#include <mitsuba/render/sdf.h>

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class SdfPlugin final : public Sdf<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sdf)
    MTS_IMPORT_TYPES()
    SdfPlugin(const Properties &props) : Base(props) { }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(SdfPlugin, Sdf)
MTS_EXPORT_PLUGIN(SdfPlugin, "Sdf plugin")

NAMESPACE_END(mitsuba)
