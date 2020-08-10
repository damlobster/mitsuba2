#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>

NAMESPACE_BEGIN(mitsuba)


// Adapt a volume to be able to serve as object texture
template <typename Float, typename Spectrum>
class Texture3D final : public Texture<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Texture, Volume)

    Texture3D(const Properties &props) : Texture(props) {
        for (auto &kv : props.objects()) {
            Volume *volume = dynamic_cast<Volume *>(kv.second.get());
            if (volume) {
                m_volume = volume;
            }
        }
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &it, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_volume->eval(it, active);
    }

    Float eval_1(const SurfaceInteraction3f &it, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_volume->eval_1(it, active);
    }

    Color3f eval_3(const SurfaceInteraction3f &it, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        auto ret = m_volume->eval_3(it, active);
        return Color3f(ret.x(), ret.y(), ret.z());
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("volume", m_volume.get());
    }

    bool is_spatially_varying() const override { return true; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Texture3D[" << std::endl
            << "  volume = " << string::indent(m_volume) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ref<Volume> m_volume;
};

MTS_IMPLEMENT_CLASS_VARIANT(Texture3D, Texture)
MTS_EXPORT_PLUGIN(Texture3D, "Texture3D texture")
NAMESPACE_END(mitsuba)
