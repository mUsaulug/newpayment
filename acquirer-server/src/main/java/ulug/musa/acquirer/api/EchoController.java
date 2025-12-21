package ulug.musa.acquirer.api;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import ulug.musa.common.security.SecurityHeaders;

@RestController
@RequestMapping("/api")
public class EchoController {

    @PostMapping(
            value = "/echo",
            consumes = MediaType.TEXT_PLAIN_VALUE,
            produces = MediaType.TEXT_PLAIN_VALUE
    )
    public String echo(
            @RequestHeader(SecurityHeaders.TERMINAL_ID) String terminalId,
            @RequestHeader(SecurityHeaders.NONCE) String nonce,
            @RequestHeader(SecurityHeaders.TIMESTAMP) String timestamp,
            @RequestHeader(SecurityHeaders.SIGNATURE) String signature,
            @RequestBody String body) {
        return body; // "selam" gönderirsen "selam" döner
    }
}
