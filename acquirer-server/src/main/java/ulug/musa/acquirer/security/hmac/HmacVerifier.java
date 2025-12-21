package ulug.musa.acquirer.security.hmac;

import ulug.musa.common.security.HmacUtil;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class HmacVerifier {

    private final String secret;
    private final Charset charset;

    public HmacVerifier(String secret, Charset charset) {
        if (secret == null || secret.isBlank()) {
            throw new IllegalArgumentException("HMAC secret boş olamaz");
        }
        this.secret = secret;
        this.charset = charset == null ? StandardCharsets.UTF_8 : charset;
    }

    public String sign(String data) {
        return HmacUtil.sign(secret, data, charset);
    }

    public boolean verify(String data, String base64Signature) {
        return HmacUtil.verify(secret, data, base64Signature, charset);
    }
}
